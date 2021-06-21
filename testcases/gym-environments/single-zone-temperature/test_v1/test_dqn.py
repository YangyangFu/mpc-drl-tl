from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

import os
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import DQNPolicy
from tianshou.utils import BasicLogger
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, VectorReplayBuffer
import torch.nn as nn
import gym_singlezone_temperature
import gym


def get_args():
    time_step = 15*60.0
    num_of_days = 7#31
    max_number_of_steps = int(num_of_days*24*60*60.0 / time_step)

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="JModelicaCSSingleZoneTemperatureEnv-v1")
    parser.add_argument('--time-step', type=float, default=time_step)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--eps-test', type=float, default=0.005)
    parser.add_argument('--eps-train', type=float, default=1.)
    parser.add_argument('--eps-train-final', type=float, default=0.05)

    parser.add_argument('--buffer-size', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--gamma', type=float, default=0.99)

    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--target-update-freq', type=int, default=40)

    parser.add_argument('--epoch', type=int, default=40)

    parser.add_argument('--step-per-epoch', type=int, default=max_number_of_steps)
    parser.add_argument('--step-per-collect', type=int, default=1)
    parser.add_argument('--update-per-step', type=float, default=1)
    parser.add_argument('--batch-size', type=int, default=64)

    parser.add_argument('--training-num', type=int, default=4)
    parser.add_argument('--test-num', type=int, default=1)

    parser.add_argument('--logdir', type=str, default='log')
    
    parser.add_argument('--device', type=str, default='cpu')# or 'cuda'
    parser.add_argument('--frames-stack', type=int, default=1)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', default=False, action='store_true',
                        help='watch the play of pre-trained policy only')
    parser.add_argument('--save-buffer-name', type=str, default='./experiments_results/his')

    parser.add_argument('--test-only', type=bool, default=False)

    return parser.parse_args()

# define external reward post-processing function
def rf(cost,penalty):
    """An external reward function

    :param cost: energy cost
    :type cost: list
    :param penalty: temperature violation penalty
    :type penalty: list
    """
    target = (cost[0] + penalty[0])/800.0
    if target > 0.0:
        clipped_target = 0.0
    elif target < -1.0:
        clipped_target = -1.0
    else:
        clipped_target = target

    return clipped_target    

def make_building_env(args):
    weather_file_path = "./USA_CA_Riverside.Muni.AP.722869_TMY3.epw"
    mass_flow_nor = [0.75]
    npre_step = 3
    simulation_start_time = 212*24*3600.0
    log_level = 7
    alpha = 100
    nActions = 37

    env = gym.make(args.task,
                   mass_flow_nor = mass_flow_nor,
                   weather_file = weather_file_path,
                   npre_step = npre_step,
                   simulation_start_time = simulation_start_time,
                   time_step = args.time_step,
                   log_level = log_level,
                   alpha = alpha,
                   nActions = nActions,
                   rf=rf)
    return env

class Net(nn.Module):
    def __init__(self, state_shape, action_shape, device):
        super().__init__()
        self.model = nn.Sequential(*[
            nn.Linear(np.prod(state_shape), 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, np.prod(action_shape))
        ])
        self.device = device
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        batch = obs.shape[0]
        
        logits = self.model(obs.view(batch, -1))
        return logits, state

import time
import tqdm
import warnings
from collections import defaultdict
from typing import Dict, Union, Callable, Optional

from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.trainer import test_episode, gather_info
from tianshou.utils import tqdm_config, MovAvg, BaseLogger, LazyLogger
def offpolicy_trainer_1(
    policy: BasePolicy,
    train_collector: Collector,
    test_collector: Collector,
    max_epoch: int,
    step_per_epoch: int,
    step_per_collect: int,
    episode_per_test: int,
    batch_size: int,
    update_per_step: Union[int, float] = 1,
    train_fn: Optional[Callable[[int, int], None]] = None,
    test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
    stop_fn: Optional[Callable[[float], bool]] = None,
    save_fn: Optional[Callable[[BasePolicy], None]] = None,
    save_checkpoint_fn: Optional[Callable[[int, int, int], None]] = None,
    resume_from_log: bool = False,
    reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    logger: BaseLogger = LazyLogger(),
    verbose: bool = True,
    test_in_train: bool = True,
) -> Dict[str, Union[float, str]]:

    if save_fn:
        warnings.warn("Please consider using save_checkpoint_fn instead of save_fn.")

    start_epoch, env_step, gradient_step = 0, 0, 0
    if resume_from_log:
        start_epoch, env_step, gradient_step = logger.restore_data()
    last_rew, last_len = 0.0, 0
    stat: Dict[str, MovAvg] = defaultdict(MovAvg)
    start_time = time.time()
    train_collector.reset_stat()
    test_collector.reset_stat()
    test_in_train = test_in_train and train_collector.policy == policy
    #test_result = test_episode(policy, test_collector, test_fn, start_epoch,
    #                           episode_per_test, logger, env_step, reward_metric)
    best_epoch = start_epoch
    best_reward, best_reward_std = -99999999, -1#test_result["rew"], test_result["rew_std"]

    for epoch in range(1 + start_epoch, 1 + max_epoch):
        # train
        policy.train()
        with tqdm.tqdm(
            total=step_per_epoch, desc=f"Epoch #{epoch}", **tqdm_config
        ) as t:
            while t.n < t.total:
                if train_fn:
                    train_fn(epoch, env_step)
                result = train_collector.collect(n_step=step_per_collect)
                if result["n/ep"] > 0 and reward_metric:
                    result["rews"] = reward_metric(result["rews"])
                env_step += int(result["n/st"])
                t.update(result["n/st"])
                logger.log_train_data(result, env_step)
                last_rew = result['rew'] if 'rew' in result else last_rew
                #print("last_rew:    ", train_collector.buffer)
                last_len = result['len'] if 'len' in result else last_len
                data = {
                    "env_step": str(env_step),
                    "rew": f"{last_rew:.2f}",
                    "len": str(int(last_len)),
                    "n/ep": str(int(result["n/ep"])),
                    "n/st": str(int(result["n/st"])),
                }
                if result["n/ep"] > 0:
                    if test_in_train and stop_fn and stop_fn(result["rew"]):
                        test_result = test_episode(
                            policy, test_collector, test_fn,
                            epoch, episode_per_test, logger, env_step)
                        if stop_fn(test_result["rew"]):
                            if save_fn:
                                save_fn(policy)
                            logger.save_data(
                                epoch, env_step, gradient_step, save_checkpoint_fn)
                            t.set_postfix(**data)
                            return gather_info(
                                start_time, train_collector, test_collector,
                                test_result["rew"], test_result["rew_std"])
                        else:
                            policy.train()
                for i in range(round(update_per_step * result["n/st"])):
                    gradient_step += 1
                    losses = policy.update(batch_size, train_collector.buffer)
                    for k in losses.keys():
                        stat[k].add(losses[k])
                        losses[k] = stat[k].get()
                        data[k] = f"{losses[k]:.3f}"
                    logger.log_update_data(losses, gradient_step)
                    t.set_postfix(**data)
            if t.n <= t.total:
                t.update()
        
        
        if save_fn:
            save_fn(policy)
        '''
        print("Setup test envs ...")
        policy.eval()
        
        test_collector.reset_stat()
        eps_test = 0.005
        policy.set_eps(eps_test)
        print("Testing agent ...")
        
                
        
        test_result = test_collector.collect(n_step=step_per_epoch)
        

        rew = test_result["rews"].mean()
        '''
        best_reward, best_reward_std = 1, 1
        #print(f'Mean reward (over {result["n/ep"]} episodes): {rew}')

    return 1#gather_info(start_time, train_collector, test_collector, best_reward, best_reward_std)

def test_dqn(args=get_args()):
    tim_env = 0.0
    tim_ctl = 0.0
    tim_learn = 0.0
    
    env = make_building_env(args)

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)


    # make environments
    train_envs = SubprocVectorEnv([lambda: make_building_env(args)
                                   for _ in range(args.training_num)])
    test_envs = SubprocVectorEnv([lambda: make_building_env(args)
                                  for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # define model
    
    net = Net(args.state_shape[0], args.action_shape, args.device).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    # define policy
    policy = DQNPolicy(net, optim, args.gamma, args.n_step,
                       target_update_freq=args.target_update_freq, reward_normalization = False, is_double=True)
    # load a previous policy
    #if args.resume_path:
    #    policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
    #    print("Loaded agent from: ", args.resume_path)
    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    buffer = VectorReplayBuffer(
        args.buffer_size, buffer_num=len(train_envs), ignore_obs_next=True,
        save_only_last_obs=False, stack_num=args.frames_stack)

    # collector
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=False)

    buffer_test = VectorReplayBuffer(
        args.step_per_epoch+100, buffer_num=len(test_envs), ignore_obs_next=True,
        save_only_last_obs=False, stack_num=args.frames_stack)
    
    test_collector = Collector(policy, test_envs, buffer_test, exploration_noise=False)

    # log
    log_path = os.path.join(args.logdir, args.task, 'dqn')
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = BasicLogger(writer)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    '''
    def stop_fn(mean_rewards):
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        else:
            return False
    '''

    def train_fn(epoch, env_step):
        # nature DQN setting, linear decay in the first 1M steps
        max_eps_steps = args.epoch * args.step_per_epoch * 0.9

        total_epoch_pass = epoch*args.step_per_epoch + env_step

        #print("observe eps:  max_eps_steps, total_epoch_pass ", max_eps_steps, total_epoch_pass)
        if env_step <= max_eps_steps:
            eps = args.eps_train - total_epoch_pass * (args.eps_train - args.eps_train_final) / max_eps_steps
        else:
            eps = args.eps_train_final
        policy.set_eps(eps)
        print('train/eps', env_step, eps)
        logger.write('train/eps', env_step, eps)

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
                ignore_obs_next=True, save_only_last_obs=False,
                stack_num=args.frames_stack)
        collector = Collector(policy, test_envs, buffer, exploration_noise=False)
        result = collector.collect(n_step=args.step_per_epoch)
        #buffer.save_hdf5(args.save_buffer_name)
        
        np.save(args.save_buffer_name+'_act.npy', buffer._meta.__dict__['act'])
        np.save(args.save_buffer_name+'_obs.npy', buffer._meta.__dict__['obs'])
        np.save(args.save_buffer_name+'_rew.npy', buffer._meta.__dict__['rew'])
        #print(buffer._meta.__dict__.keys())
        rew = result["rews"].mean()
        print(f'Mean reward (over {result["n/ep"]} episodes): {rew}')
    
    if not args.test_only:
        # test train_collector and start filling replay buffer
        train_collector.collect(n_step=args.batch_size * args.training_num)
        # trainer
        
        result = offpolicy_trainer_1(
            policy = policy, train_collector = train_collector, test_collector = test_collector, max_epoch = args.epoch,
            step_per_epoch = args.step_per_epoch, step_per_collect = args.step_per_collect, episode_per_test = args.test_num,
            batch_size = args.batch_size, train_fn=train_fn, test_fn=test_fn,
            #stop_fn=stop_fn, 
            save_fn=save_fn, logger=logger,
            update_per_step=args.update_per_step, test_in_train=False)
        '''

        result = offpolicy_trainer_1(
            policy = policy, train_collector = train_collector, test_collector = test_collector, max_epoch = 1,
            step_per_epoch = 100, step_per_collect = 1, episode_per_test = 1,
            batch_size = 64, train_fn=train_fn, test_fn=test_fn,
            #stop_fn=stop_fn, 
            save_fn=save_fn, logger=logger,
            update_per_step=args.update_per_step, test_in_train=False)
        '''
        #pprint.pprint(result)

        watch()
      
    if args.test_only:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", os.path.join(log_path, 'policy.pth'))
        watch()

    '''
    for k in range(args.epoch):
        
        for i in range(int(args.step_per_epoch)):  # total step
            max_eps_steps = args.step_per_epoch * 0.9
            if env_step <= max_eps_steps:
                eps = args.eps_train - env_step * (args.eps_train - args.eps_train_final) / max_eps_steps
            else:
                eps = args.eps_train_final
            policy.set_eps(eps)

            collect_result = train_collector.collect(n_step=10)

            losses = policy.update(64, train_collector.buffer)
        
        train_collector.reset_env()
    '''

if __name__ == '__main__':
    import sys
    print("Python version")
    print (sys.version)
    test_dqn(get_args())