#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint

import gym_singlezone_jmodelica
import gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.policy import NPGPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
import time



def get_args(folder):

    time_step = 15*60.0
    num_of_days = 7#31
    max_number_of_steps = int(num_of_days*24*60*60.0 / time_step)

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='JModelicaCSSingleZoneEnv-v2')
    parser.add_argument('--time-step', type=float, default=time_step)
    parser.add_argument('--step-per-epoch', type=int, default=max_number_of_steps)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=4096)#!!!!!!!!!!!
    parser.add_argument(
        '--hidden-sizes', type=int, nargs='*', default=[256,256,256,256]
    )  # baselines [32, 32]
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=500)#!!!!!!!!!!!!100
    parser.add_argument('--step-per-collect', type=int, default=96*4)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    # batch-size >> step-per-collect means calculating all data in one singe forward.
    parser.add_argument('--batch-size', type=int, default=99999)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    # npg special
    parser.add_argument('--rew-norm', type=int, default=True)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--bound-action-method', type=str, default="clip")
    parser.add_argument('--lr-decay', type=int, default=True)
    parser.add_argument('--logdir', type=str, default='log_npg')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--norm-adv', type=int, default=1)
    parser.add_argument('--optim-critic-iters', type=int, default=20)
    parser.add_argument('--actor-step-size', type=float, default=0.1)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', default=False, action='store_true',
                        help='watch the play of pre-trained policy only')
    parser.add_argument('--save-buffer-name', type=str, default=folder)
    return parser.parse_args()

def make_building_env(args):
    weather_file_path = "./USA_CA_Riverside.Muni.AP.722869_TMY3.epw"
    mass_flow_nor = [0.75]
    npre_step = 3
    simulation_start_time = 212*24*3600.0
    simulation_end_time = simulation_start_time + args.step_per_epoch*args.time_step
    log_level = 0
    alpha = 1

    def rw_func(cost, penalty):
        if ( not hasattr(rw_func,'x')  ):
            rw_func.x = 0
            rw_func.y = 0

        cost = cost[0]
        penalty = penalty[0]

        if rw_func.x > cost:
            rw_func.x = cost
        if rw_func.y > penalty:
            rw_func.y = penalty

        #print("rw_func-cost-min=", rw_func.x, ". penalty-min=", rw_func.y)

        res = (penalty * 500.0 + cost*5e4)/1000.0#!!!!!!!!!!!!!!!!!!
        
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
                   rf = rw_func)
    return env


def test_npg(args):
    env = make_building_env(args)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
    # train_envs = gym.make(args.task)
    train_envs = SubprocVectorEnv(
        [lambda: make_building_env(args) for _ in range(args.training_num)], #norm_obs=True
    )
    # test_envs = gym.make(args.task)
    test_envs = SubprocVectorEnv(
        [lambda: make_building_env(args) for _ in range(args.test_num)],
        #norm_obs=True,
        obs_rms=train_envs.obs_rms,
        update_obs_rms=False
    )

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net_a = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        device=args.device
    )
    actor = ActorProb(
        net_a,
        args.action_shape,
        max_action=args.max_action,
        unbounded=True,#!!!????
        device=args.device
    ).to(args.device)
    net_c = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        device=args.device
    )
    critic = Critic(net_c, device=args.device).to(args.device)
    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    # do last policy layer scaling, this will make initial actions have (close to)
    # 0 mean and std, and will help boost performances,
    # see https://arxiv.org/abs/2006.05990, Fig.24 for details
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)

    optim = torch.optim.Adam(critic.parameters(), lr=args.lr)
    lr_scheduler = None
    if args.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(
            args.step_per_epoch / args.step_per_collect
        ) * args.epoch

        lr_scheduler = LambdaLR(
            optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
        )

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = NPGPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        reward_normalization=args.rew_norm,
        action_scaling=True,
        action_bound_method=args.bound_action_method,
        lr_scheduler=lr_scheduler,
        action_space=env.action_space,
        advantage_normalization=args.norm_adv,
        optim_critic_iters=args.optim_critic_iters,
        actor_step_size=args.actor_step_size
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
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_npg'
    log_path = os.path.join(args.logdir, args.task, 'npg', log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer, update_interval=100, train_interval=100)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    if not args.watch:
        # trainer
        result = onpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.repeat_per_collect,
            args.test_num,
            args.batch_size,
            step_per_collect=args.step_per_collect,
            save_fn=save_fn,
            logger=logger,
            test_in_train=False
        )
        pprint.pprint(result)

    def watch():
        print("Setup test envs ...")
        policy.eval()
        test_envs.seed(args.seed)

        print("Testing agent ...")
        buffer = VectorReplayBuffer(args.step_per_epoch+1, len(test_envs))

        collector = Collector(policy, test_envs, buffer, exploration_noise=False)
        result = collector.collect(n_step=args.step_per_epoch)
        
        np.save(args.save_buffer_name+'/his_act.npy', buffer._meta.__dict__['act'])
        np.save(args.save_buffer_name+'/his_obs.npy', buffer._meta.__dict__['obs'])
        np.save(args.save_buffer_name+'/his_rew.npy', buffer._meta.__dict__['rew'])
        #print(buffer._meta.__dict__.keys())
        rew = result["rews"].mean()
        print(f'Mean reward (over {result["n/ep"]} episodes): {rew}')

    watch()


if __name__ == '__main__':
    folder='./npg_results'
    if not os.path.exists(folder):
        os.mkdir(folder)

    start = time.time()
    print("Begin time {}".format(start))

    test_npg(get_args(folder))

    end = time.time()
    print("Total execution time {:.2f} seconds".format(end-start))