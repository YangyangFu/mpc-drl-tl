from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

import gym
import gym_singlezone_jmodelica
import tianshou as ts
print(ts.__version__)

env = gym.make("JModelicaCSSingleZoneEnv-price-v1",
                mass_flow_nor=0.75,
                weather_file='USA_CA_Riverside.Muni.AP.722869_TMY3.epw',
                npre_step=3,
                simulation_start_time=3600*24,
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
print(env.tau, env.simulation_start_time)
print(env._get_action_space(),env.nActions)
print(env.alpha)
print("JModelicaCSSingleZoneEnv-price-v1 is successfully installed!!" )