import argparse
import os
import pprint

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import ShmemVectorEnv
from tianshou.policy import IQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.discrete import ImplicitQuantileNetwork
import torch.nn as nn
import gym

def make_building_env(args):
    import gym_singlezone_jmodelica

    weather_file_path = "USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw"
    mass_flow_nor = [0.55]
    npre_step = 3
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
                   mass_flow_nor = mass_flow_nor,
                   weather_file = weather_file_path,
                   npre_step = npre_step,
                   simulation_start_time = simulation_start_time,
                   simulation_end_time = simulation_end_time,
                   time_step = args.time_step,
                   log_level = log_level,
                   alpha = alpha,
                   nActions = nActions,
                   rf = rw_func)
    return env

class DQN(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        state_shape,
        action_shape,
        nlayers,
        device,
        features_only: bool = False,
    ):
        super().__init__()
        self.action_num = np.prod(action_shape)
        self.device = device
        sequences = [nn.Linear(np.prod(state_shape), 256), nn.ReLU(inplace=True)]
        for i in range(nlayers):
            sequences.append(nn.Linear(256, 256))
            sequences.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*sequences)

        self.output_dim = 256
        if not features_only:
            self.net = nn.Sequential(
                self.net, nn.Linear(self.output_dim, 512), nn.ReLU(inplace=True),
                nn.Linear(512, np.prod(action_shape))
            )
            self.output_dim = np.prod(action_shape)


    def forward(self, obs, state=None, info={}):
        r"""Mapping: x -> Z(x, \*)."""
        
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        batch = obs.shape[0]
        
        logits = self.net(obs.view(batch, -1))

        return logits, state

def test_iqn(args):
    env = make_building_env(args)

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
   
    # make environments
    train_envs = ShmemVectorEnv([lambda: make_building_env(args) for _ in range(args.training_num)])
    test_envs = ShmemVectorEnv([lambda: make_building_env(args) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # define model
    feature_net = DQN(args.state_shape, args.action_shape, args.n_hidden_layers, args.device, features_only=True)
    net = ImplicitQuantileNetwork(
        feature_net,
        args.action_shape,
        args.hidden_sizes,
        num_cosines=args.num_cosines,
        device=args.device
    ).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    # define policy
    policy = IQNPolicy(
        net,
        optim,
        args.gamma,
        args.sample_size,
        args.online_sample_size,
        args.target_sample_size,
        args.n_step,
        target_update_freq=args.target_update_freq
    ).to(args.device)
    
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)
    
    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    buffer = VectorReplayBuffer(
        args.buffer_size,
        buffer_num=len(train_envs),
        #ignore_obs_next=True,
        #save_only_last_obs=True,
        stack_num=args.frames_stack
    )
    # collector
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # log
    log_path = os.path.join(args.logdir, args.task, 'iqn')
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
        #print('train/eps', env_step, eps)
        #logger.write('train/eps', env_step, eps)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    # watch agent's performance
    def watch():
        print("Setup test envs ...")
        policy.eval()
        policy.set_eps(args.eps_test)
        test_envs.seed(args.seed)

        print("Testing agent ...")
        buffer = VectorReplayBuffer(
                args.step_per_epoch+1, buffer_num=len(test_envs),
                ignore_obs_next=True, save_only_last_obs=False,#!!!!!!!!!!!!
                stack_num=args.frames_stack)
        collector = Collector(policy, test_envs, buffer, exploration_noise=False)
        result = collector.collect(n_step=args.step_per_epoch)
        #buffer.save_hdf5(args.save_buffer_name)
        
        np.save(os.path.join(args.logdir, args.task,'his_act.npy'), buffer._meta.__dict__['act'])
        np.save(os.path.join(args.logdir, args.task,'his_obs.npy'), buffer._meta.__dict__['obs'])
        np.save(os.path.join(args.logdir, args.task,'his_rew.npy'), buffer._meta.__dict__['rew'])
        #print(buffer._meta.__dict__.keys())
        rew = result["rews"].mean()
        print(f'Mean reward (over {result["n/ep"]} episodes): {rew}')


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
        watch()
    
    if args.test_only:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", os.path.join(log_path, 'policy.pth'))
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
        test_iqn(args)

        # a fake traing score to stop current simulation based on searched parameters
        reporter(timesteps_total=args.step_per_epoch)

if __name__ == '__main__':
    import ray
    from ray import tune

    time_step = 15*60.0
    num_of_days = 1#31
    max_number_of_steps = int(num_of_days*24*60*60.0 / time_step)

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="JModelicaCSSingleZoneEnv-v1")
    parser.add_argument('--time-step', type=float, default=time_step)
    parser.add_argument('--step-per-epoch', type=int, default=max_number_of_steps)
    parser.add_argument('--seed', type=int, default=16)
    parser.add_argument('--eps-test', type=float, default=0.005)
    parser.add_argument('--eps-train', type=float, default=1.)
    parser.add_argument('--eps-train-final', type=float, default=0.05)

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--sample-size', type=int, default=32)
    parser.add_argument('--online-sample-size', type=int, default=8)
    parser.add_argument('--target-sample-size', type=int, default=8)
    parser.add_argument('--num-cosines', type=int, default=128)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[512])
    parser.add_argument('--n-step', type=int, default=3)#!!!!!!3
    parser.add_argument('--target-update-freq', type=int, default=300)

    parser.add_argument('--step-per-collect', type=int, default=10)#!!!!!!!10
    parser.add_argument('--update-per-step', type=float, default=0.1)#!!!!!!!!!!!1
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log_iqn')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--frames-stack', type=int, default=1)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--test-only', type=bool, default=False)

    # tunnable parameters
    parser.add_argument('--buffer-size', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=0.01)  # 0.0001
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--weight-energy', type=float, default=100.)
    parser.add_argument('--weight-temp', type=float, default=1.)
    parser.add_argument('--n-hidden-layers', type=int, default=3)

    args = parser.parse_args()
    test_iqn(args)
    """
    # Define Ray tuning experiments
    tune.register_trainable("iqn", trainable_function)
    ray.init()

    # Run tuning
    tune.run_experiments({
        'iqn_tuning': {
            "run": "iqn",
            "stop": {"timesteps_total": args.step_per_epoch},
            "config": {
                "epoch": tune.grid_search([1]),
                "weight_energy": tune.grid_search([10, 100]),
                "lr": tune.grid_search([1e-03, 3e-03, 0.01]),
                "batch_size": tune.grid_search([32, 64, 128]),
                "n_hidden_layer": tune.grid_search([3]),
                "buffer_size": tune.grid_search([50000])
            },
            "local_dir": "/mnt/shared",
        }
    })
"""
"""
    tune.run_experiments({
            'iqn_tuning':{
                "run": "iqn",
                "stop": {"timesteps_total":args.step_per_epoch},
                "config":{
                    "epoch": tune.grid_search([200]),
                    "weight_energy": tune.grid_search([0.1, 1, 10., 100., 1000., 10000.]),
                    "lr": tune.grid_search([3e-04, 1e-04, 1e-03]),
                    "batch_size": tune.grid_search([32, 64, 128]),
                    "n_hidden_layer": tune.grid_search([3, 4]),
                    "buffer_size":tune.grid_search([20000, 50000, 100000])
                    },
                "local_dir":"/mnt/shared",
            }
    })
"""
