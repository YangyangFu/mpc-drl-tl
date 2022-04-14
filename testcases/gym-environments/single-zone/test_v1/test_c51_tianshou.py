import argparse
import os
import pprint

import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.policy import C51Policy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger

import torch.nn as nn
import gym_singlezone_jmodelica
import gym

def make_building_env(args):
    import gym_singlezone_jmodelica

    weather_file_path = "USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw"
    mass_flow_nor = [0.55]
    n_next_steps = 4
    simulation_start_time = 201*24*3600.0
    simulation_end_time = simulation_start_time + args.step_per_epoch*args.time_step
    log_level = 7
    alpha = 1
    nActions = 51
    weight_energy = args.weight_energy  # 5.e4
    weight_temp = args.weight_temp  # 500.

    def rw_func(cost, penalty):
        if (not hasattr(rw_func, 'x')):
            rw_func.x = 0
            rw_func.y = 0

        cost = cost[0]
        penalty = penalty[0]

        if rw_func.x > cost:
            rw_func.x = cost
        if rw_func.y > penalty:
            rw_func.y = penalty

        print("rw_func-cost-min=", rw_func.x, ". penalty-min=", rw_func.y)
        res = penalty * weight_temp + cost*weight_energy

        return res

    env = gym.make(args.task,
                   mass_flow_nor=mass_flow_nor,
                   weather_file=weather_file_path,
                   n_next_steps=n_next_steps,
                   simulation_start_time=simulation_start_time,
                   simulation_end_time=simulation_end_time,
                   time_step=args.time_step,
                   log_level=log_level,
                   alpha=alpha,
                   nActions=nActions,
                   rf=rw_func)
    return env

class C51(nn.Module):
    """Reference: A distributional perspective on reinforcement learning.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        state_shape,
        action_shape,
        nlayers,
        num_atoms,
        device
    ):
        super().__init__()
        self.action_num = np.prod(action_shape)
        self.device = device
        sequences = [nn.Linear(np.prod(state_shape), 256),
                     nn.ReLU(inplace=True)]
        for i in range(nlayers):
            sequences.append(nn.Linear(256, 256))
            sequences.append(nn.ReLU(inplace=True))
        sequences.append(nn.Linear(256, self.action_num * num_atoms))
        self.net = nn.Sequential(*sequences)

        self.output_dim = self.action_num * num_atoms

        self.num_atoms = num_atoms

    def forward(self, x, state=None, info={}):
        r"""Mapping: x -> Z(x, \*)."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        batch = x.shape[0]
        
        x = self.net(x.view(batch, -1))
        x = x.view(-1, self.num_atoms).softmax(dim=-1)
        x = x.view(-1, self.action_num, self.num_atoms)
        return x, state


def test_c51(args):
    env = make_building_env(args)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    
    # make environments
    train_envs = SubprocVectorEnv(
            [lambda: make_building_env(args) for _ in range(args.training_num)], 
            norm_obs=True)
    test_envs = SubprocVectorEnv(
            [lambda: make_building_env(args) for _ in range(args.test_num)], 
            norm_obs=True, 
            obs_rms=train_envs.obs_rms, 
            update_obs_rms=False)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    
    # define model
    net = C51(args.state_shape, args.action_shape, args.n_hidden_layers, args.num_atoms, args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    # define policy
    policy = C51Policy(
        net,
        optim,
        args.gamma,
        args.num_atoms,
        args.v_min,
        args.v_max,
        args.n_step,
        target_update_freq=args.target_update_freq,
    ).to(args.device)
 
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", os.path.join(log_path, 'policy.pth'))
    
    # collector
    buffer = VectorReplayBuffer(
        args.buffer_size,
        buffer_num=len(train_envs),
        ignore_obs_next=True,
        stack_num=args.frames_stack
    )
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=False)

    # log
    log_path = os.path.join(args.logdir, args.task)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def train_fn(epoch, env_step):

        # nature DQN setting, linear decay in the first 1M steps
        max_eps_steps = int(args.epoch * args.step_per_epoch * 0.9)

        #print("observe eps:  max_eps_steps, total_epoch_pass ", max_eps_steps, total_epoch_pass)
        if env_step <= max_eps_steps:
            eps = args.eps_train - env_step * (args.eps_train - args.eps_train_final) / max_eps_steps
        else:
            eps = args.eps_train_final
        policy.set_eps(eps)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    if not args.test_only:

        # test train_collector and start filling replay buffer
        train_collector.collect(n_step=args.batch_size * args.training_num)
        # trainer
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.step_per_collect,
            args.test_num,
            args.batch_size,
            train_fn=train_fn,
            test_fn=test_fn,
            save_fn=save_fn,
            logger=logger,
            update_per_step=args.update_per_step,
            test_in_train=False
        )

        pprint.pprint(result)

    # Lets watch its performance for the final run
    def watch():
        print("Setup test envs ...")
        policy.eval()
        policy.set_eps(args.eps_test)

        buffer = VectorReplayBuffer(
            args.step_per_epoch,
            buffer_num=len(test_envs),
            ignore_obs_next=True,
            save_only_last_obs=False,
            stack_num=args.frames_stack)
        collector = Collector(policy, test_envs, buffer)
        result = collector.collect(n_step=args.step_per_epoch)

        # get obs mean and var
        obs_mean = test_envs.obs_rms.mean
        obs_var = test_envs.obs_rms.var
        print(obs_mean)
        print(obs_var)
        # the observations and action may be normalized depending on training setting
        np.save(os.path.join(args.logdir, args.task, 'his_act.npy'),
                buffer._meta.__dict__['act'])
        np.save(os.path.join(args.logdir, args.task, 'his_obs.npy'),
                buffer._meta.__dict__['obs'])
        np.save(os.path.join(args.logdir, args.task, 'his_rew.npy'),
                buffer._meta.__dict__['rew'])
        np.save(os.path.join(args.logdir, args.task, 'obs_mean.npy'), obs_mean)
        np.save(os.path.join(args.logdir, args.task, 'obs_var.npy'), obs_var)
        rew = result["rews"].mean()
        print(f'Mean reward (over {result["n/ep"]} episodes): {rew}')

    watch()

# added hyperparameter tuning scripting using Ray.tune
def trainable_function(config, reporter):
    while True:
        args.epoch = config['epoch']
        args.weight_energy = config['weight_energy']
        args.lr = config['lr']
        args.batch_size = config['batch_size']
        args.n_hidden_layer = config['n_hidden_layer']
        args.buffer_size = config['buffer_size']
        test_c51(args)

        # a fake traing score to stop current simulation based on searched parameters
        reporter(timesteps_total=args.step_per_epoch)


if __name__ == '__main__':
    import ray
    from ray import tune

    time_step = 15*60.0
    num_of_days = 1  # 31
    max_number_of_steps = int(num_of_days*24*60*60.0 / time_step)

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str,
                        default="JModelicaCSSingleZoneEnv-v1")
    parser.add_argument('--time-step', type=float, default=time_step)
    parser.add_argument('--step-per-epoch', type=int,
                        default=max_number_of_steps)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eps-test', type=float, default=0.005)
    parser.add_argument('--eps-train', type=float, default=1.)
    parser.add_argument('--eps-train-final', type=float, default=0.05)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--num-atoms', type=int, default=201)
    parser.add_argument('--v-min', type=float, default=-600)  # -10.
    parser.add_argument('--v-max', type=float, default=0)  # 10.#600
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--target-update-freq', type=int, default=300)
    parser.add_argument('--step-per-collect', type=int, default=1)
    parser.add_argument('--update-per-step', type=float, default=1)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log_c51')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--frames-stack', type=int, default=1)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--test-only', type=bool, default=False)

    # tunable parameters
    parser.add_argument('--weight-energy', type=float, default=100.)
    parser.add_argument('--weight-temp', type=float, default=1.)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--n-hidden-layers', type=int, default=3)
    parser.add_argument('--buffer-size', type=int, default=50000)

    args = parser.parse_args()

    # Define Ray tuning experiments
    tune.register_trainable("c51", trainable_function)
    ray.init()

    # Run tuning
    tune.run_experiments({
        'c51_tuning': {
            "run": "c51",
            "stop": {"timesteps_total": args.step_per_epoch},
            "config": {
                "epoch": tune.grid_search([1]),
                "weight_energy": tune.grid_search([10]),
                "lr": tune.grid_search([3e-04]),
                "batch_size": tune.grid_search([32, 64, 128]),
                "n_hidden_layer": tune.grid_search([3]),
                "buffer_size": tune.grid_search([50000])
            },
            "local_dir": "/mnt/shared",
        }
    })
    """
    tune.run_experiments({
        'c51_tuning': {
            "run": "c51",
            "stop": {"timesteps_total": args.step_per_epoch},
            "config": {
                "epoch": tune.grid_search([400]),
                "weight_energy": tune.grid_search([0.1, 1, 10, 100.]),
                "lr": tune.grid_search([1e-04, 3e-04, 1e-03]),
                "batch_size": tune.grid_search([32, 64, 128]),
                "n_hidden_layer": tune.grid_search([3]),
                "buffer_size": tune.grid_search([20000, 50000, 100000])
            },
            "local_dir": "/mnt/shared",
        }
    })
    """
