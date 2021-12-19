import os
import gym_fivezoneair_jmodelica
import gym
import torch
import pprint
import datetime
import argparse
import numpy as np
import json
import time
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Independent, Normal

from tianshou.policy import PPOPolicy
from tianshou.utils import BasicLogger
from tianshou.env import SubprocVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer


def get_args(folder):

    time_step = 15*60.0
    num_of_days = 7#31
    max_number_of_steps = int(num_of_days*24*60*60.0 / time_step)

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='JModelicaCSFiveZoneAirEnv-v2')
    parser.add_argument('--time-step', type=float, default=time_step)
    parser.add_argument('--step-per-epoch', type=int, default=max_number_of_steps)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=4096)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128,128,128,128])#!!!
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-collect', type=int, default=96*4) #2048
    parser.add_argument('--repeat-per-collect', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    # ppo special
    parser.add_argument('--rew-norm', type=int, default=True)
    # In theory, `vf-coef` will not make any difference if using Adam optimizer.
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--bound-action-method', type=str, default="clip")
    parser.add_argument('--lr-decay', type=int, default=True)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=0)
    parser.add_argument('--norm-adv', type=int, default=0)
    parser.add_argument('--recompute-adv', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log_ppo')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', default=False, action='store_true',
                        help='watch the play of pre-trained policy only')
    parser.add_argument('--save-buffer-name', type=str, default=folder)
    return parser.parse_args()

def make_building_env(args):
    weather_file_path = "./USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw"
    npre_step = 0
    simulation_start_time = 159*24*3600.0
    simulation_end_time = simulation_start_time + args.step_per_epoch*args.time_step
    log_level = 0
    alpha = 1

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
        res = (penalty * 5000 + cost*500) / 1000
        
        return res

    env = gym.make(args.task,
                   weather_file = weather_file_path,
                   npre_step = npre_step,
                   simulation_start_time = simulation_start_time,
                   simulation_end_time = simulation_end_time,
                   time_step = args.time_step,
                   log_level = log_level,
                   alpha = alpha,
                   rf = rw_func)
    return env

import time
from collections import defaultdict
from typing import Callable, Dict, Optional, Union

import numpy as np
import tqdm

from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.trainer import gather_info, test_episode
from tianshou.utils import BaseLogger, LazyLogger, MovAvg, tqdm_config


def onpolicy_trainer1(
    args: Dict,
    test_envs: SubprocVectorEnv,
    policy: BasePolicy,
    train_collector: Collector,
    test_collector: Collector,
    max_epoch: int,
    step_per_epoch: int,
    repeat_per_collect: int,
    episode_per_test: int,
    batch_size: int,
    step_per_collect: Optional[int] = None,
    episode_per_collect: Optional[int] = None,
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
    """A wrapper for on-policy trainer procedure.
    The "step" in trainer means an environment step (a.k.a. transition).
    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
    :param Collector train_collector: the collector used for training.
    :param Collector test_collector: the collector used for testing.
    :param int max_epoch: the maximum number of epochs for training. The training
        process might be finished before reaching ``max_epoch`` if ``stop_fn`` is set.
    :param int step_per_epoch: the number of transitions collected per epoch.
    :param int repeat_per_collect: the number of repeat time for policy learning, for
        example, set it to 2 means the policy needs to learn each given batch data
        twice.
    :param int episode_per_test: the number of episodes for one policy evaluation.
    :param int batch_size: the batch size of sample data, which is going to feed in the
        policy network.
    :param int step_per_collect: the number of transitions the collector would collect
        before the network update, i.e., trainer will collect "step_per_collect"
        transitions and do some policy network update repeatedly in each epoch.
    :param int episode_per_collect: the number of episodes the collector would collect
        before the network update, i.e., trainer will collect "episode_per_collect"
        episodes and do some policy network update repeatedly in each epoch.
    :param function train_fn: a hook called at the beginning of training in each epoch.
        It can be used to perform custom additional operations, with the signature ``f(
        num_epoch: int, step_idx: int) -> None``.
    :param function test_fn: a hook called at the beginning of testing in each epoch.
        It can be used to perform custom additional operations, with the signature ``f(
        num_epoch: int, step_idx: int) -> None``.
    :param function save_fn: a hook called when the undiscounted average mean reward in
        evaluation phase gets better, with the signature ``f(policy: BasePolicy) ->
        None``.
    :param function save_checkpoint_fn: a function to save training process, with the
        signature ``f(epoch: int, env_step: int, gradient_step: int) -> None``; you can
        save whatever you want.
    :param bool resume_from_log: resume env_step/gradient_step and other metadata from
        existing tensorboard log. Default to False.
    :param function stop_fn: a function with signature ``f(mean_rewards: float) ->
        bool``, receives the average undiscounted returns of the testing result,
        returns a boolean which indicates whether reaching the goal.
    :param function reward_metric: a function with signature ``f(rewards: np.ndarray
        with shape (num_episode, agent_num)) -> np.ndarray with shape (num_episode,)``,
        used in multi-agent RL. We need to return a single scalar for each episode's
        result to monitor training in the multi-agent RL setting. This function
        specifies what is the desired metric, e.g., the reward of agent 1 or the
        average reward over all agents.
    :param BaseLogger logger: A logger that logs statistics during
        training/testing/updating. Default to a logger that doesn't log anything.
    :param bool verbose: whether to print the information. Default to True.
    :param bool test_in_train: whether to test in the training phase. Default to True.
    :return: See :func:`~tianshou.trainer.gather_info`.
    .. note::
        Only either one of step_per_collect and episode_per_collect can be specified.
    """
    start_epoch, env_step, gradient_step = 0, 0, 0
    if resume_from_log:
        start_epoch, env_step, gradient_step = logger.restore_data()
    last_rew, last_len = 0.0, 0
    stat: Dict[str, MovAvg] = defaultdict(MovAvg)
    start_time = time.time()
    train_collector.reset_stat()
    test_collector.reset_stat()
    test_in_train = test_in_train and train_collector.policy == policy
    test_result = test_episode(
        policy, test_collector, test_fn, start_epoch, episode_per_test, logger,
        env_step, reward_metric
    )
    best_epoch = start_epoch
    best_reward, best_reward_std = test_result["rew"], test_result["rew_std"]

    # acts=[]
    # obss=[]
    # rews=[]
    
    for epoch in range(1 + start_epoch, 1 + max_epoch):
        # train
        policy.train()
        with tqdm.tqdm(
            total=step_per_epoch, desc=f"Epoch #{epoch}", **tqdm_config
        ) as t:
            while t.n < t.total:
                if train_fn:
                    train_fn(epoch, env_step)
                result = train_collector.collect(
                    n_step=step_per_collect, n_episode=episode_per_collect
                )
                print ("result", result)
                
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
                if result["n/ep"] > 0:
                    if test_in_train and stop_fn and stop_fn(result["rew"]):
                        test_result = test_episode(
                            policy, test_collector, test_fn, epoch, episode_per_test,
                            logger, env_step
                        )
                        if stop_fn(test_result["rew"]):
                            if save_fn:
                                save_fn(policy)
                            logger.save_data(
                                epoch, env_step, gradient_step, save_checkpoint_fn
                            )
                            t.set_postfix(**data)
                            return gather_info(
                                start_time, train_collector, test_collector,
                                test_result["rew"], test_result["rew_std"]
                            )
                        else:
                            policy.train()
                losses = policy.update(
                    0,
                    train_collector.buffer,
                    batch_size=batch_size,
                    repeat=repeat_per_collect
                )
                train_collector.reset_buffer(keep_statistics=True)
                step = max(
                    [1] + [len(v) for v in losses.values() if isinstance(v, list)]
                )
                gradient_step += step
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

        # # watch for each episode to save data
        # print("Setup test envs ...")
        # policy.eval()
        # #policy.set_eps(args.eps_test)
        # test_envs.seed(args.seed)

        # print("Testing agent ...")
        # buffer = VectorReplayBuffer(args.step_per_epoch+1, len(test_envs))

        # collector = Collector(policy, test_envs, buffer, exploration_noise=False)
        # result = collector.collect(n_step=args.step_per_epoch)

        # #buffer.save_hdf5(args.save_buffer_name)
        # act=buffer._meta.__dict__['act']
        # obs=buffer._meta.__dict__['obs']
        # rew=buffer._meta.__dict__['rew']
        # acts.append(act)
        # obss.append(obs)
        # rews.append(rew)
        # #print(buffer._meta.__dict__.keys())
        # rew = result["rews"].mean()
        # print(f'Mean reward (over {result["n/ep"]} episodes): {rew}')

    # np.save(args.save_buffer_name+'/his_act.npy', np.array(acts))
    # np.save(args.save_buffer_name+'/his_obs.npy', np.array(obss))
    # np.save(args.save_buffer_name+'/his_rew.npy', np.array(rews))

    return 1
def test_ppo(args):
    env = make_building_env(args)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low),
          np.max(env.action_space.high))

    train_envs = SubprocVectorEnv([lambda: make_building_env(args) for _ in range(args.training_num)])
    #test_envs = make_building_env(args)
    test_envs = SubprocVectorEnv([lambda: make_building_env(args) for _ in range(args.test_num)])

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes,
                activation=nn.Tanh, device=args.device)
    actor = ActorProb(net_a, args.action_shape, max_action=args.max_action,
                      unbounded=False, device=args.device).to(args.device)
    net_c = Net(args.state_shape, hidden_sizes=args.hidden_sizes,
                activation=nn.Tanh, device=args.device)
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
    
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.lr)

    lr_scheduler = None
    if args.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(args.step_per_epoch / args.step_per_collect) * args.epoch
        lr_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = PPOPolicy(actor, critic, optim, dist, discount_factor=args.gamma,
                       gae_lambda=args.gae_lambda, max_grad_norm=args.max_grad_norm,
                       vf_coef=args.vf_coef, ent_coef=args.ent_coef,
                       reward_normalization=args.rew_norm, action_scaling=True,
                       action_bound_method=args.bound_action_method,
                       lr_scheduler=lr_scheduler, action_space=env.action_space,
                       eps_clip=args.eps_clip, value_clip=args.value_clip,
                       dual_clip=args.dual_clip, advantage_normalization=args.norm_adv,
                       recompute_advantage=args.recompute_adv)

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    train_collector = Collector(policy, train_envs, buffer)#, exploration_noise=False)
    test_collector = Collector(policy, test_envs)
    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_ppo'
    log_path = os.path.join(args.logdir, args.task, 'ppo', log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = BasicLogger(writer, update_interval=100, train_interval=100)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    if not args.watch:
        # cutomized trainer
        result = onpolicy_trainer1(args, test_envs,
            policy, train_collector, test_collector, args.epoch, args.step_per_epoch,
            args.repeat_per_collect, args.test_num, args.batch_size,
            step_per_collect=args.step_per_collect, save_fn=save_fn, logger=logger,
            test_in_train=False)
        # trainer
        #result = onpolicy_trainer(
        #    policy, train_collector, test_collector, args.epoch, args.step_per_epoch,
        #    args.repeat_per_collect, args.test_num, args.batch_size,
        #    step_per_collect=args.step_per_collect, save_fn=save_fn, logger=logger,
        #    test_in_train=False)
        pprint.pprint(result)

    def watch():
        print("Setup test envs ...")
        policy.eval()
        test_envs.seed(args.seed)

        print("Testing agent ...")
        buffer = VectorReplayBuffer(args.step_per_epoch+1, len(test_envs))

        collector = Collector(policy, test_envs, buffer)#, exploration_noise=False)
        result = collector.collect(n_step=args.step_per_epoch)
        
        np.save(args.save_buffer_name+'/his_act_final.npy', buffer._meta.__dict__['act'])
        np.save(args.save_buffer_name+'/his_obs_final.npy', buffer._meta.__dict__['obs'])
        np.save(args.save_buffer_name+'/his_rew_final.npy', buffer._meta.__dict__['rew'])
        #print(buffer._meta.__dict__.keys())
        rew = result["rews"].mean()
        print(f'Mean reward (over {result["n/ep"]} episodes): {rew}')

    watch()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)    
    folder='./ppo_results'
    if not os.path.exists(folder):
        os.mkdir(folder)

    args = get_args(folder)
    start = time.time()
    print("Begin time {}".format(start))

    test_ppo(args)

    end = time.time()
    print("Total execution time {:.2f} seconds".format(end-start))

    # save training statistics
    statistics = {"training time":end-start, 
                    "episode": args.epoch}
    with open('statistics.json', 'w') as fp:
        json.dump(statistics,fp)