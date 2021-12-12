import os
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import DQNPolicy
from tianshou.utils import BasicLogger, TensorboardLogger
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, VectorReplayBuffer
import torch.nn as nn
import gym_fivezoneair_jmodelica
import gym


def get_args(folder="experiment_results"):
    time_step = 15*60.0
    num_of_days = 7#31
    max_number_of_steps = int(num_of_days*24*60*60.0 / time_step)

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="JModelicaCSFiveZoneAirEnv-v1")
    parser.add_argument('--time-step', type=float, default=time_step)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--eps-test', type=float, default=0.005)
    parser.add_argument('--eps-train', type=float, default=1.)
    parser.add_argument('--eps-train-final', type=float, default=0.05)

    parser.add_argument('--buffer-size', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=0.0003)#0.0003!!!!!!!!!!!!!!!!!!!!!
    parser.add_argument('--gamma', type=float, default=0.99)

    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--target-update-freq', type=int, default=100)

    parser.add_argument('--epoch', type=int, default=100)

    parser.add_argument('--step-per-epoch', type=int, default=max_number_of_steps)
    parser.add_argument('--step-per-collect', type=int, default=1)
    parser.add_argument('--update-per-step', type=float, default=1)
    parser.add_argument('--batch-size', type=int, default=128)

    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)

    parser.add_argument('--logdir', type=str, default='log_ddqn')
    
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--frames-stack', type=int, default=1)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', default=False, action='store_true',
                        help='watch the play of pre-trained policy only')
    parser.add_argument('--save-buffer-name', type=str, default=folder)

    parser.add_argument('--test-only', type=bool, default=False)


    return parser.parse_args()


def make_building_env(args):
    weather_file_path = "./USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw"
    npre_step = 0
    simulation_start_time = 204*24*3600.0
    simulation_end_time = simulation_start_time + args.step_per_epoch*args.time_step
    log_level = 7
    alpha = 1
    nActions = 51

    def rw_func(cost, penalty):
        if ( not hasattr(rw_func,'x')  ):
            rw_func.x = 0
            rw_func.y = 0

        print(cost, penalty)
        #res = cost + penalty 
        cost = cost[0]
        penalty = penalty[0]
        
        print(cost, penalty)
        
        if rw_func.x > cost:
            rw_func.x = cost
        if rw_func.y > penalty:
            rw_func.y = penalty

        print("rw_func-cost-min=", rw_func.x, ". penalty-min=", rw_func.y)
        #res = penalty * 10.0
        #res = penalty * 300.0 + cost*1e4
        res = penalty * 500.0 + cost*500
        
        return res

    env = gym.make(args.task,
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

class Net(nn.Module):
    def __init__(self, state_shape, action_shape, device):
        super().__init__()
        self.model = nn.Sequential(*[
            nn.Linear(np.prod(state_shape), 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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

    history_reward_epoch=[]
    history_loss=[]
    history_action=(train_collector.buffer._meta.__dict__['act'][:batch_size]).tolist()
    history_state=(train_collector.buffer._meta.__dict__['obs'][:batch_size]).tolist()
    history_reward=(train_collector.buffer._meta.__dict__['rew'][:batch_size]).tolist()

    for epoch in range(1 + start_epoch, 1 + max_epoch):
        # train
        policy.train()
        train_collector.reset_env()
        with tqdm.tqdm(
            total=step_per_epoch, desc=f"Epoch #{epoch}", **tqdm_config
        ) as t:
            while t.n < t.total:
                if train_fn:
                    train_fn(epoch, env_step)
                result = train_collector.collect(n_step=step_per_collect)
                print(result)
                
                #Save the history results for each epoch
                
                history_action.append(train_collector.buffer._meta.__dict__['act'][(batch_size+env_step) % len(train_collector.buffer)])
                history_state.append(train_collector.buffer._meta.__dict__['obs'][(batch_size+env_step) % len(train_collector.buffer)])
                history_reward.append(train_collector.buffer._meta.__dict__['rew'][(batch_size+env_step) % len(train_collector.buffer)])
                
                if result["n/ep"] > 0 and reward_metric:
                    result["rews"] = reward_metric(result["rews"])
                env_step += int(result["n/st"])
                t.update(result["n/st"])
                logger.log_train_data(result, env_step)
                last_rew = result['rew'] if 'rew' in result else last_rew
                last_len = result['len'] if 'len' in result else last_len
                data = {
                    "env_step": str(env_step),
                    "rew": f"{last_rew:.2f}",
                    "len": str(int(last_len)),
                    "n/ep": str(int(result["n/ep"])),
                    "n/st": str(int(result["n/st"])),
                }
                #print("last_rew:    ", train_collector.buffer)
                for i in range(round(update_per_step * result["n/st"])):
                    gradient_step += 1
                    print("gradient_step: ",gradient_step)
                    losses = policy.update(batch_size, train_collector.buffer)
                    print("losses:    ", losses)
                    for k in losses.keys():
                        stat[k].add(losses[k])
                        losses[k] = stat[k].get()
                        data[k] = f"{losses[k]:.3f}"
                    logger.log_update_data(losses, gradient_step)
                    t.set_postfix(**data)
                history_loss.append(list(losses.values()))
            if t.n <= t.total:
                t.update()

            history_reward_epoch.append(result["rews"])
            
        if save_fn:
            save_fn(policy)

        

    return history_reward, history_state, history_action, history_loss, history_reward_epoch

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
    print(args.state_shape)
    net = Net(args.state_shape, args.action_shape, args.device).to(args.device)
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
        args.buffer_size, buffer_num=len(train_envs), ignore_obs_next=True)
    
    # collector
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)




    buffer_test = VectorReplayBuffer(
        args.step_per_epoch+100, buffer_num=len(test_envs), ignore_obs_next=True)
    
    test_collector = Collector(policy, test_envs, buffer_test, exploration_noise=True)

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
        #total_epoch_pass = env_step
        print("env_step: ", env_step)
        print("observe eps:  current epoch, step in current epoch, total_epoch_pass,  max_eps_steps", epoch, env_step % args.step_per_epoch, total_epoch_pass, max_eps_steps)
        if env_step <= max_eps_steps:
            eps = args.eps_train - total_epoch_pass * (args.eps_train - args.eps_train_final) / max_eps_steps
        # observe eps:  max_eps_steps, total_epoch_pass  60480.0 103477 train/eps env_step eps 51733 -0.625382771164021
        else:
            eps = args.eps_train_final
        policy.set_eps(eps)
        print('train/eps', env_step, eps)
        logger.write("save/eps", env_step, {"save/eps": eps})

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
        print (result) 
        #{'n/ep': 1, 'n/st': 672, 'rews': array([-1032753.50984556]), 'lens': array([672]), 'idxs': array([0]), 'rew': -1032753.5098455583, 'len': 672.0, 'rew_std': 0.0, 'len_std': 0.0}
        
        np.save(args.save_buffer_name+'/his_act_final.npy', buffer._meta.__dict__['act'])
        np.save(args.save_buffer_name+'/his_obs_final.npy', buffer._meta.__dict__['obs'])
        np.save(args.save_buffer_name+'/his_rew_final.npy', buffer._meta.__dict__['rew'])
        #print(buffer._meta.__dict__.keys())
        
        rew = result["rews"].mean()
        print(f'Mean reward (over {result["n/ep"]} episodes): {rew}')
    

    if not args.test_only:

        # test train_collector and start filling replay buffer
        train_collector.collect(n_step=args.batch_size * args.training_num)
        # trainer
        
        history_reward, history_state, history_action, history_loss, history_reward_epoch = offpolicy_trainer_1(
            policy = policy, train_collector = train_collector, test_collector = test_collector, max_epoch = args.epoch,
            step_per_epoch = args.step_per_epoch, step_per_collect = args.step_per_collect, episode_per_test = args.test_num,
            batch_size = args.batch_size, train_fn=train_fn, test_fn=test_fn,
            #stop_fn=stop_fn, 
            save_fn=save_fn, logger=logger,
            update_per_step=args.update_per_step, test_in_train=False)

        np.save(args.save_buffer_name+'/rew_per_epoch.npy', history_reward_epoch)

        #output is arrays of size 50000
        np.save(args.save_buffer_name+'/his_act_hist.npy', history_action)
        np.save(args.save_buffer_name+'/his_obs_hist.npy', history_state)
        np.save(args.save_buffer_name+'/his_rew_hist.npy', history_reward)
        np.save(args.save_buffer_name+'/his_loss_hist.npy', history_loss)
        
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
    folder='./dqn_results'
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    test_dqn(get_args(folder=folder))
