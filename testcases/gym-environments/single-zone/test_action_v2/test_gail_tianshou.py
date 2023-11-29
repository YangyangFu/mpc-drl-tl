import argparse
import os
import pickle
import pprint

import gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

from tianshou.data import Collector, VectorReplayBuffer, ReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv, VectorEnvNormObs
from tianshou.policy import GAILPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic



class NoRewardEnv(gym.RewardWrapper):
    """sets the reward to 0.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        """Set reward to 0."""
        return np.zeros_like(reward)

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

    def rw_func(cost, penalty, delta_action): # try making 0 reward for gail training by setting rw_func = 0 - NO NEED TO DO THIS
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
        # set the reward to 0 for gail training - NO NEED TO DO THIS
        # res = 0.0*res
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

def test_gail(args):
    # read expert replay buffer
    if os.path.exists(args.expert_data_task) and os.path.isfile(args.expert_data_task):
        if args.expert_data_task.endswith(".hdf5"):
            expert_buffer = VectorReplayBuffer.load_hdf5(args.expert_data_task)
        else:
            expert_buffer = pickle.load(open(args.expert_data_task, "rb"))
            print("Load expert buffer from %s" % args.expert_data_task)
            print("Buffer size:", expert_buffer.maxsize)
            print("Keys included in expert buffer", expert_buffer._meta.__dict__.keys())
    else:
        raise ValueError("Expert data path is not valid.")

    env = make_building_env(args)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), 
          np.max(env.action_space.high))

    # make environments
    train_envs = SubprocVectorEnv(
            [lambda: make_building_env(args) for _ in range(args.training_num)])
    train_envs = VectorEnvNormObs(train_envs)
    test_envs = SubprocVectorEnv(
            [lambda: make_building_env(args) for _ in range(args.test_num)])
    test_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
    test_envs.set_obs_rms(train_envs.get_obs_rms())

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
                device=args.device)
    actor = ActorProb(
                net_a, 
                args.action_shape, 
                max_action=args.max_action, 
                unbounded=True, #whether to apply tanh activation on final logits
                device=args.device
                ).to(args.device)
    net_c = Net(
                args.state_shape, 
                hidden_sizes=args.hidden_sizes,
                activation=nn.Tanh, 
                device=args.device)
    critic = Critic(
                net_c,
                device=args.device
                ).to(args.device)
    actor_critic = ActorCritic(actor, critic)
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
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    # discriminator
    net_d = Net(
            args.state_shape,
            action_shape=args.action_shape,
            hidden_sizes=args.hidden_sizes,
            activation=torch.nn.Tanh,
            device=args.device,
            concat=True,
    )
    disc_net = Critic(
        net_d,
        device=args.device
    ).to(args.device)
    for m in disc_net.modules():
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    disc_optim = torch.optim.Adam(disc_net.parameters(), lr=args.disc_lr)

    lr_scheduler = None
    if args.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(args.step_per_epoch / args.step_per_collect) * args.epoch
        lr_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)

    # replace DiagGuassian with Independent(Normal) which is equivalent
    # pass *logits to be consistent with policy.forward
    def dist(*logits):
        return Independent(Normal(*logits), 1)

    print(env.action_space)

    policy = GAILPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        expert_buffer=expert_buffer,
        disc_net=disc_net,
        disc_optim=disc_optim,
        disc_update_num=args.disc_update_num,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        action_scaling=False, # weahter to map action from range [-1,1] to range[action-space.low, action_spaces.high]
        action_bound_method=args.bound_action_method,
        lr_scheduler=lr_scheduler,
        action_space=env.action_space,
        eps_clip=args.eps_clip,
        value_clip=args.value_clip,
        dual_clip=args.dual_clip,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
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
    logger = TensorboardLogger(writer, save_interval=args.save_interval)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    if not args.test_only:
        # trainer
        result = onpolicy_trainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            repeat_per_collect=args.repeat_per_collect,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            # episode_per_collect=args.episode_per_collect, #!!!!!!!
            step_per_collect=args.step_per_collect, #!!!!!!!
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=False,
        )
        pprint.pprint(result)

    def watch():
        print("Setup test envs ...")
        policy.eval()
        test_envs.seed(args.seed)

        print("Testing agent ...")
        buffer = VectorReplayBuffer(args.step_per_epoch, len(test_envs))

        collector = Collector(policy, test_envs, buffer, exploration_noise=False)
        result = collector.collect(n_step=args.step_per_epoch)

        # Save the buffer data as a pickle file
        # with open(args.save_buffer_name, "wb") as f:
        #     # pickle.dump(transition_data, f)
        #     pickle.dump(buffer, f)
        # print(f"Data collected and saved to {args.save_buffer_name}")

        # get obs mean and var
        obs_mean = test_envs.obs_rms.mean
        obs_var = test_envs.obs_rms.var
        print(obs_mean)
        print(obs_var)
        # save data
        np.save(os.path.join(args.logdir, args.task, 'his_act.npy'),policy.map_action(buffer.act)) 
        np.save(os.path.join(args.logdir, args.task, 'his_obs.npy'),buffer.obs)
        np.save(os.path.join(args.logdir, args.task, 'his_rew.npy'),buffer.rew)
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
        args.hidden_sizes=[256]*args.n_hidden_layers  # baselines [32, 32]
        args.step_per_collect = config['step_per_collect']
        args.seed = config['seed']
        args.disc_lr = config['disc_lr']
        test_gail(args)

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
    parser.add_argument('--repeat-per-collect', type=int, default=10)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    # ppo special
    parser.add_argument('--rew-norm', type=int, default=True)
    # In theory, `vf-coef` will not make any difference if using Adam optimizer.
    parser.add_argument('--vf-coef', type=float, default=0.25)
    parser.add_argument('--ent-coef', type=float, default=0.0) # 0.001???
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    # bound action to [-1,1] using different methods. empty means no bounding
    parser.add_argument('--bound-action-method', type=str, default="clip")
    parser.add_argument('--lr-decay', type=int, default=True)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=0)
    parser.add_argument('--norm-adv', type=int, default=0)
    parser.add_argument('--recompute-adv', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log_gail')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--test-only', type=bool, default=False)
    # GAIL special
    parser.add_argument('--disc-lr', type=float, default=2.5e-5)
    parser.add_argument("--disc-update-num", type=int, default=2)
    parser.add_argument("--expert-data-task", type=str, default=r'/mnt/shared/expert_SAC_JModelicaCSSingleZoneEnv-action-v2.pkl') # Change to your own path

    parser.add_argument("--save-interval", type=int, default=1)

    # tunable parameters
    parser.add_argument('--weight-energy', type=float, default= 100.)   
    parser.add_argument('--weight-temp', type=float, default= 1.)  
    parser.add_argument('--weight-action', type=float, default=10.) 
    parser.add_argument('--lr', type=float, default=0.0003) #0.0003!!!!!!!!!!!!!!!!!!!!!
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-hidden-layers', type=int, default=3)
    parser.add_argument('--buffer-size', type=int, default=4096)
    parser.add_argument('--save-buffer-name', type=str, default='expert_GAIL_JModelicaCSSingleZoneEnv-action-v2.pkl')
    args = parser.parse_args()

    # Define Ray tuning experiments
    tune.register_trainable("gail", trainable_function)
    ray.init()

    # Run tuning
    tune.run_experiments({
        'gail_tuning': {
            "run": "gail",
            "stop": {"timesteps_total": args.step_per_epoch},
            "config": {
                "epoch": tune.grid_search([100]), # try default 500 for the first run
                "weight_energy": tune.grid_search([100.]),
                "lr": tune.grid_search([3e-04]), #[1e-03]
                "disc_lr": tune.grid_search([2.5e-05]),
                "batch_size": tune.grid_search([256]),
                "n_hidden_layers": tune.grid_search([3]),
                "buffer_size": tune.grid_search([4096]),
                "step_per_collect": tune.grid_search([672*4]), #[256, 512]
                "eps_clip": tune.grid_search([0.2]),
                "seed": tune.grid_search([5])
            },
            "local_dir": "/mnt/shared",
        }
    })