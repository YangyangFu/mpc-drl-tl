# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 23:20:11 2023

@author: Mingyue Guo
"""

import os
import gym
import torch
import pickle
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import DQNPolicy
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer, PrioritizedReplayBuffer
from tianshou.data import VectorReplayBuffer
from tianshou.env import SubprocVectorEnv

def get_args():
    '''
    max_epoch：最大允许的训练轮数，有可能没训练完这么多轮就会停止（因为满足了 stop_fn 的条件）

    step_per_epoch：每个epoch要更新多少次策略网络

    collect_per_step：每次更新前要收集多少帧与环境的交互数据。上面的代码参数意思是，每收集10帧进行一次网络更新

    episode_per_test：每次测试的时候花几个rollout进行测试

    batch_size：每次策略计算的时候批量处理多少数据

    train_fn：在每个epoch训练之前被调用的函数，输入的是当前第几轮epoch和当前用于训练的env一共step了多少次。上面的代码意味着，在每次训练前将epsilon设置成0.1

    test_fn：在每个epoch测试之前被调用的函数，输入的是当前第几轮epoch和当前用于训练的env一共step了多少次。上面的代码意味着，在每次测试前将epsilon设置成0.05

    stop_fn：停止条件，输入是当前平均总奖励回报（the average undiscounted returns），返回是否要停止训练

    writer：天授支持 TensorBoard，可以像下面这样初始化：

    :return:
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='CartPole-v0')  # 环境名
    parser.add_argument('--seed', type=int, default=1626)  # 随机种子
    parser.add_argument('--eps-test', type=float, default=0.05)  # 贪婪策略的比例
    parser.add_argument('--eps-train', type=float, default=0.1)  # 贪婪策略的比例
    parser.add_argument('--buffer-size', type=int, default=20000)  # 回放池大小
    parser.add_argument('--lr', type=float, default=1e-3)  # 学习率
    parser.add_argument('--gamma', type=float, default=0.9)  # 衰减率
    parser.add_argument('--n-step', type=int, default=3)  # 要向前看的步数
    parser.add_argument('--target-update-freq', type=int, default=320)  # 目标网络的更新频率，每隔freq次更新一次，0为不使用目标网络
    parser.add_argument('--epoch', type=int, default=10)  # 世代
    parser.add_argument('--step-per-epoch', type=int, default=1000)  # 每个世代策略网络更新的次数
    parser.add_argument('--collect-per-step', type=int, default=10)  # 网络更新之前收集的帧数
    parser.add_argument('--batch-size', type=int, default=64)  # 神经网络批训练大小
    parser.add_argument('--hidden-sizes', type=int,
                        nargs='*', default=[128, 128, 128, 128])  # 隐藏层尺寸
    parser.add_argument('--training-num', type=int, default=8)  # 学习环境数量
    parser.add_argument('--test-num', type=int, default=100)  # 测试环境数量
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--prioritized-replay',
                        action="store_true", default=False)  # 优先重播
    parser.add_argument('--alpha', type=float, default=0.6)  # 经验池参数，每轮所有样本进行指数变换的常数
    parser.add_argument('--beta', type=float, default=0.4)  # 经验池参数，重要抽样权重的常数，内含一个公式的化简，详细看源码
    parser.add_argument(
        '--save-buffer-name', type=str,
        default="./expert_DQN_CartPole-v0.pkl")
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def test_dqn(args=get_args()):
    env = gym.make(args.task)  # 构建env
    # 状态纬度
    args.state_shape = env.observation_space.shape or env.observation_space.n
    # 行动数量
    args.action_shape = env.action_space.shape or env.action_space.n
    # train_envs = gym.make(args.task)

    # 构建envs，dummyvectorenv运用for实现 subpro运用多进程实现
    # you can also use tianshou.env.SubprocVectorEnv
    # train_envs = DummyVectorEnv(
    #     [lambda: gym.make(args.task) for _ in range(args.training_num)])
    # # test_envs = gym.make(args.task)
    # test_envs = DummyVectorEnv(
    #     [lambda: gym.make(args.task) for _ in range(args.test_num)])
    

    train_envs = SubprocVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)], 
        norm_obs=True)
    test_envs = SubprocVectorEnv(
            [lambda: gym.make(args.task) for _ in range(args.test_num)],
            norm_obs=True, 
            obs_rms=train_envs.obs_rms, 
            update_obs_rms=False)
    # seed 设置随机种子 方便复现
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # Q_param = V_param = {"hidden_sizes": [128]}
    # model
    # 构建神经网络模型，Net是已经被定义好的类
    net = Net(args.state_shape, args.action_shape,
              hidden_sizes=args.hidden_sizes, device=args.device,
              # dueling=(Q_param, V_param),
              ).to(args.device)
    # 优化器
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    # 策略
    policy = DQNPolicy(
        net, optim, args.gamma, args.n_step,
        target_update_freq=args.target_update_freq)
    # buffer 缓存回放
    if args.prioritized_replay:
        buf = PrioritizedReplayBuffer(
            args.buffer_size, alpha=args.alpha, beta=args.beta)
    else:
        buf = VectorReplayBuffer(args.buffer_size,
                                 buffer_num=len(train_envs), 
                                 ignore_obs_next=True)
    # collector 收集器，主要控制环境与策略的交互
    train_collector = Collector(policy, train_envs, buf)
    test_collector = Collector(policy, test_envs)
    # policy.set_eps(1)
    # batchsize是神经网络训练一轮的参数，所以必须要一次性输入batchsize个经验
    # 也就是 需要隔batchsize步进行训练
    # 先进行一轮采集，防止经验池为空，采集的参数在后面会被清空，只留下经验池
    train_collector.collect(n_step=args.batch_size, no_grad=False)
    # log
    log_path = os.path.join(args.logdir, args.task, 'dqn')
    # writer = SummaryWriter(log_path)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    # 停止条件 平均回报大于阈值
    def stop_fn(mean_rewards):
        return mean_rewards >= env.spec.reward_threshold

    # 学习前调用的函数
    # 在每个epoch训练之前被调用的函数，输入的是当前第几轮epoch和当前用于训练的env一共step了多少次。
    # 此处为了实现根据一个世代中的迭代次数改变eps(贪婪策略的比例)
    def train_fn(epoch, env_step):
        # eps annnealing, just a demo
        if env_step <= 10000:
            policy.set_eps(args.eps_train)
        elif env_step <= 50000:
            eps = args.eps_train - (env_step - 10000) / \
                  40000 * (0.9 * args.eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * args.eps_train)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    # trainer 开始学习
    # 异策略
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_num,
        args.batch_size, train_fn=train_fn, test_fn=test_fn,
        stop_fn=stop_fn, save_fn=save_fn
        # , writer=writer
        )

    # assert stop_fn(result['best_reward'])

    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        env = gym.make(args.task)
        policy.eval()
        policy.set_eps(args.eps_test)
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')

    # save buffer in pickle format, for imitation learning unittest
    buf = VectorReplayBuffer(args.buffer_size,
                                buffer_num=len(test_envs), 
                                ignore_obs_next=True)
    collector = Collector(policy, test_envs, buf)
    #与环境进行交互，具体为每走一步就判断是否有环境结束，如果环境结束，则将环境所走的步数加入总步数
    #n_step为其最少的步数，即大于则收集结束
    collector.collect(n_step=args.buffer_size)

    pickle.dump(buf, open(args.save_buffer_name, "wb"))


def test_pdqn(args=get_args()):
    args.prioritized_replay = True
    args.gamma = .95
    args.seed = 1
    test_dqn(args)


if __name__ == '__main__':
    test_dqn(get_args())
