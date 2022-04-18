import os
import gym
import torch
import pprint
import datetime
import argparse
import numpy as np
import time
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Independent, Normal

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic

class Wrapper(gym.Wrapper):
    """Env wrapper for reward scale when using sac"""

    def __init__(self, env, reward_scale=10):
        super().__init__(env)
        self.reward_scale = reward_scale
    
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        # scale a reward
        return obs, self.reward_scale*rew, done, info

def make_building_env(args):
    import gym_singlezone_jmodelica
    
    weather_file_path = "USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw"
    mass_flow_nor = [0.55]
    n_next_steps = 4
    n_prev_steps = 4
    simulation_start_time = 201*24*3600.0
    simulation_end_time = simulation_start_time + args.step_per_epoch*args.time_step
    log_level = 0
    alpha = 1
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

        #print("rw_func-cost-min=", rw_func.x, ". penalty-min=", rw_func.y)
        #res = (penalty * 500.0 + cost*5e4)/1000.0#!!!!!!!!!!!!!!!!!!
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
                   rf = rw_func,
                   n_prev_steps=n_prev_steps)
    return env

def test_sac(args):
    env = Wrapper(make_building_env(args), reward_scale=args.reward_scale)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low),
          np.max(env.action_space.high))

    train_envs = SubprocVectorEnv(
            [lambda: Wrapper(make_building_env(args),reward_scale=args.reward_scale) for _ in range(args.training_num)],
            norm_obs=True)
    test_envs = SubprocVectorEnv(
            [lambda: Wrapper(make_building_env(args), reward_scale=1) for _ in range(args.test_num)], 
            norm_obs=True, 
            obs_rms=train_envs.obs_rms, 
            update_obs_rms=False)

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net_a = Net(
                args.state_shape, 
                hidden_sizes=args.hidden_sizes,
                device=args.device)
    actor = ActorProb(
                net_a, 
                args.action_shape, 
                max_action=args.max_action,
                unbounded=True, #whether to apply tanh activation on final logits
                conditioned_sigma=True,
                device=args.device
                ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device
    )
    net_c2 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device
    )

    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)


    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
        action_space=env.action_space
    )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    # log
    log_path = os.path.join(args.logdir, args.task)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    if not args.test_only:
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
            save_fn=save_fn,
            logger=logger,
            update_per_step=args.update_per_step,
            test_in_train=False)
        pprint.pprint(result)

    def watch():
        print("Setup test envs ...")
        policy.eval()
        test_envs.seed(args.seed)

        print("Testing agent ...")
        buffer = VectorReplayBuffer(args.step_per_epoch, len(test_envs))

        collector = Collector(policy, test_envs, buffer, exploration_noise=False)
        result = collector.collect(n_step=args.step_per_epoch)

        # get obs mean and var
        obs_mean = test_envs.obs_rms.mean
        obs_var = test_envs.obs_rms.var
        print(obs_mean)
        print(obs_var)
        # save data
        np.save(os.path.join(args.logdir, args.task, 'his_act.npy'),buffer._meta.__dict__['act'])
        np.save(os.path.join(args.logdir, args.task, 'his_obs.npy'),buffer._meta.__dict__['obs'])
        np.save(os.path.join(args.logdir, args.task, 'his_rew.npy'),buffer._meta.__dict__['rew'])
        np.save(os.path.join(args.logdir, args.task, 'obs_mean.npy'),obs_mean)
        np.save(os.path.join(args.logdir, args.task, 'obs_var.npy'),obs_var)
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
        args.n_hidden_layers = config['n_hidden_layers']
        args.buffer_size = config['buffer_size']
        args.reward_scale = config['reward_scale']
        args.hidden_sizes = [256]*args.n_hidden_layers  # baselines [32, 32]
        args.actor_lr = args.lr
        args.critic_lr = args.lr
        test_sac(args)

        # a fake traing score to stop current simulation based on searched parameters
        reporter(timesteps_total=args.step_per_epoch)

if __name__ == '__main__':
    
    import ray
    from ray import tune

    time_step = 15*60.0
    num_of_days = 7  # 31
    max_number_of_steps = int(num_of_days*24*60*60.0 / time_step)

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='JModelicaCSSingleZoneEnv-action-v2')
    parser.add_argument('--time-step', type=float, default=time_step)
    parser.add_argument('--step-per-epoch', type=int, default=max_number_of_steps)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--step-per-collect', type=int, default=1)#!!!!!!!!!!!!!
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--update-per-step', type=int, default=1)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument("--start-timesteps", type=int, default=10000)
    # sac special
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--auto-alpha', default=False, action='store_true')
    parser.add_argument('--alpha-lr', type=float, default=3e-4)
    parser.add_argument('--reward-scale', type=float, default=10)
        
    parser.add_argument('--logdir', type=str, default='log_sac')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--test-only', type=bool, default=False)

    # tunable parameters
    parser.add_argument('--weight-energy', type=float, default= 100.)   
    parser.add_argument('--weight-temp', type=float, default= 1.)  
    parser.add_argument('--weight-action', type=float, default=10.) 
    parser.add_argument('--lr', type=float, default=1e-03)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-hidden-layers', type=int, default=3)
    parser.add_argument('--buffer-size', type=int, default=200000)

    args = parser.parse_args()

    # Define Ray tuning experiments
    tune.register_trainable("sac", trainable_function)
    ray.init()

    # Run tuning
    tune.run_experiments({
        'sac_tuning': {
            "run": "sac",
            "stop": {"timesteps_total": args.step_per_epoch},
            "config": {
                "epoch": tune.grid_search([500]),
                "weight_energy": tune.grid_search([100.]),
                "lr": tune.grid_search([1e-04, 1e-03, 3e-03]),
                "batch_size": tune.grid_search([64]),
                "n_hidden_layers": tune.grid_search([3]),
                "buffer_size": tune.grid_search([100000]),
                "reward_scale": tune.grid_search([1, 10, 30])
            },
            "local_dir": "/mnt/shared",
        }
    })
