from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

import gym
import gym_singlezone_jmodelica
import tianshou as ts

env = gym.make("JModelicaCSSingleZoneEnv-v1",
                mass_flow_nor=0.75,
                weather_file='USA_CA_Riverside.Muni.AP.722869_TMY3.epw',
                npre_step=5,
                simulation_start_time=3600*24.,
                simulation_end_time=3600*24*2.,
                time_step=15*60.,
                log_level=7,
                fmu_result_handling='memory',
                fmu_result_ncp=100.,
                filter_flag=True,
                alpha=200,
                nActions=11)
states = env.reset()
n_outputs = env.observation_space.shape[0]
print(states)
print(n_outputs)
print(env.tau, env.simulation_start_time, env.simulation_end_time)
print(env._get_action_space(),env.nActions)
print(env.alpha)
print(ts.__version__)
print("JModelicaCSSingleZoneEnv-v1 is successfully installed!!" )
## =========================================
##          Testing stablebaseline3 installation
##============================================
import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    env_id = "CartPole-v1"
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)

    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=25000)

    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        #env.render()
print("\n stablebaseline3 is successfully installed ! \n")