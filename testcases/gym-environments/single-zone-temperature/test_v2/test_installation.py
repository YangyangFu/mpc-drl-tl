from __future__ import print_function
from __future__ import absolute_import, division

import gym
import gym_singlezone_temperature
from gym import envs

env = gym.make("JModelicaCSSingleZoneTemperatureEnv-v2",
                mass_flow_nor=[0.75],
                weather_file='USA_CA_Riverside.Muni.AP.722869_TMY3.epw',
                npre_step=3,
                simulation_start_time=3600*24,
                time_step=15*60.,
                log_level=7,
                fmu_result_handling='memory',
                fmu_result_ncp=100.,
                filter_flag=True,
                alpha=200,
                min_action=12.,
                max_action=30.,)
states = env.reset()
n_outputs = env.observation_space.shape[0]
print(states)
print(env.tau, env.simulation_start_time)
print(n_outputs)
print(env.alpha)
print(env.min_action,env.max_action)