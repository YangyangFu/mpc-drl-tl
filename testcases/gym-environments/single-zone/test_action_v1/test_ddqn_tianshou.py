import os
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import DQNPolicy
from tianshou.utils import TensorboardLogger
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, VectorReplayBuffer
import torch.nn as nn
import gym

def make_building_env(args):
    import gym_singlezone_jmodelica

    weather_file_path = "USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw"
    mass_flow_nor = [0.55]
    n_next_steps = 4
    n_prev_steps = 4
    simulation_start_time = 201*24*3600.0
    simulation_end_time = simulation_start_time + args.step_per_epoch*args.time_step
    log_level = 7
    alpha = 1
    nActions = 51
    weight_energy = args.weight_energy #5.e4
    weight_temp = args.weight_temp #500.
    weight_action = args.weight_action

    def rw_func(cost, penalty, delta_action):
        if ( not hasattr(rw_func,'x')  ):
            rw_func.x = 0
            rw_func.y = 0

        cost = cost[0]
        penalty = penalty[0]
        delta_action = delta_action[0]
        if rw_func.x > cost:
            rw_func.x = cost
        if rw_func.y > penalty:
            rw_func.y = penalty

        print("rw_func-cost-min=", rw_func.x, ". penalty-min=", rw_func.y)
        res = -penalty*penalty * weight_temp + cost*weight_energy - delta_action*delta_action*weight_action
        
        return res

    env = gym.make(args.task,
                   mass_flow_nor = mass_flow_nor,
                   weather_file = weather_file_path,
                   n_next_steps = n_next_steps,
                   simulation_start_time = simulation_start_time,
                   simulation_end_time = simulation_end_time,
                   time_step = args.time_step,
                   log_level = log_level,
                   alpha = alpha,
                   nActions = nActions,
                   rf = rw_func,
                   n_prev_steps = n_prev_steps)
    return env

class Net(nn.Module):
    def __init__(self, state_shape, action_shape, nlayers,device):
        super().__init__()
        # define ann
        sequences = [nn.Linear(np.prod(state_shape), 256), nn.ReLU(inplace=True)]
        for i in range(nlayers):
            sequences.append(nn.Linear(256, 256))
            sequences.append(nn.ReLU(inplace=True))
        sequences.append(nn.Linear(256, np.prod(action_shape)))
        self.model = nn.Sequential(*sequences)
        # device
        self.device = device

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        batch = obs.shape[0]
        
        logits = self.model(obs.view(batch, -1))
        return logits, state


def test_dqn(args):

    env = make_building_env(args)

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)

    # make environments: normalize the obs space
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
    print(args.state_shape)
    net = Net(args.state_shape, args.action_shape, args.n_hidden_layers, args.device).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    # define policy
    policy = DQNPolicy(net, 
                    optim, 
                    args.gamma, 
                    args.n_step,
                    target_update_freq=args.target_update_freq, 
                    reward_normalization = False, 
                    is_double=True)
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", os.path.join(log_path, 'policy.pth'))

    # collector
    buffer = VectorReplayBuffer(
            args.buffer_size, 
            buffer_num=len(train_envs), 
            ignore_obs_next=True)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)

    # log
    log_path = os.path.join(args.logdir, args.task)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)
    
    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))
    
    def train_fn(epoch, env_step):
        # nature DQN setting, linear decay in the first 1M steps
        #max_eps_steps = int(args.epoch * args.step_per_epoch * 0.9) # this will not help speedup learning with large epoch
        max_eps_steps =200*96*7 # linear decay in the first about 100000 steps. 
        #print("observe eps:  max_eps_steps, total_epoch_pass ", max_eps_steps, total_epoch_pass)
        if env_step <= max_eps_steps:
            eps = args.eps_train - env_step * (args.eps_train - args.eps_train_final) / max_eps_steps
        else:
            eps = args.eps_train_final
        policy.set_eps(eps)
        print('train/eps', env_step, eps)
        print("=========================")
        #logger.write('train/eps', env_step, eps)
    
    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    if not args.test_only:
        # start filling replay buffer by collecting data at the beginning
        train_collector.collect(n_step=args.batch_size * args.training_num, random=False)
        # trainer
        
        result = offpolicy_trainer(
            policy = policy, 
            train_collector = train_collector, 
            test_collector = test_collector, # if none, no testing will be performed
            max_epoch = args.epoch,
            step_per_epoch = args.step_per_epoch, 
            step_per_collect = args.step_per_collect, 
            episode_per_test = args.test_num,
            batch_size = args.batch_size, 
            train_fn = train_fn, 
            test_fn = test_fn,
            #stop_fn=stop_fn, 
            save_fn = save_fn, 
            logger = logger,
            update_per_step = args.update_per_step, 
            test_in_train = False)
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
        np.save(os.path.join(args.logdir, args.task, 'obs_mean.npy'),
                obs_mean)
        np.save(os.path.join(args.logdir, args.task, 'obs_var.npy'),
                obs_var)
        rew = result["rews"].mean()
        print(f'Mean reward (over {result["n/ep"]} episodes): {rew}')
    watch()

# added hyperparameter tuning scripting using Ray.tune
def trainable_function(config, reporter):
    while True:
        args.epoch = config['epoch']
        args.weight_action = config['weight_action']
        args.lr = config['lr']
        args.batch_size = config['batch_size']
        args.n_hidden_layers = config['n_hidden_layer']
        args.buffer_size = config['buffer_size']
        args.seed = config['seed']
        test_dqn(args)

        # a fake traing score to stop current simulation based on searched parameters
        reporter(timesteps_total=args.step_per_epoch)

if __name__ == '__main__':
    import ray 
    from ray import tune

    time_step = 15*60.0
    num_of_days = 7#31
    max_number_of_steps = int(num_of_days*24*60*60.0 / time_step)

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="JModelicaCSSingleZoneEnv-action-v1")
    parser.add_argument('--time-step', type=float, default=time_step)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--eps-test', type=float, default=0.005)
    parser.add_argument('--eps-train', type=float, default=1.)
    parser.add_argument('--eps-train-final', type=float, default=0.05)

    parser.add_argument('--gamma', type=float, default=0.99)

    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--target-update-freq', type=int, default=96)

    parser.add_argument('--step-per-epoch', type=int, default=max_number_of_steps)
    parser.add_argument('--step-per-collect', type=int, default=1)
    parser.add_argument('--update-per-step', type=float, default=1)

    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1) 
    parser.add_argument('--logdir', type=str, default='log_ddqn')
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--frames-stack', type=int, default=1)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--test-only', type=bool, default=False)

    # tunable parameters
    parser.add_argument('--weight-energy', type=float, default= 100.)   
    parser.add_argument('--weight-temp', type=float, default= 1.)   
    parser.add_argument('--weight-action', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.0003) #0.0003!!!!!!!!!!!!!!!!!!!!!
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--n-hidden-layers', type=int, default=3)
    parser.add_argument('--buffer-size', type=int, default=50000)

    args = parser.parse_args()

    # Define Ray tuning experiments
    tune.register_trainable("ddqn", trainable_function)
    ray.init()

    # Run tuning
    tune.run_experiments({
        'ddqn_tuning': {
            "run": "ddqn",
            "stop": {"timesteps_total": args.step_per_epoch},
            "config": {
                "epoch": tune.grid_search([500]),
                "weight_action": tune.grid_search([10]),
                "lr": tune.grid_search([1e-04]),
                "batch_size": tune.grid_search([256]),
                "n_hidden_layers": tune.grid_search([3]),
                "buffer_size": tune.grid_search([4096*3]),
                "seed":tune.grid_search([0, 1, 2, 3, 4, 5])
            },
            "local_dir": "/mnt/shared",
        }
    })
