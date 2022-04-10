from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

import gym
import gym_singlezone_jmodelica
import tianshou as ts

env = gym.make("JModelicaCSSingleZoneEnv-price-v2",
                mass_flow_nor=0.75,
               weather_file='USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw',
                n_next_steps=3,
                simulation_start_time=3600*24.,
                simulation_end_time=3600*24*2.,
                time_step=15*60.,
                log_level=7,
                fmu_result_handling='memory',
                fmu_result_ncp=100.,
                filter_flag=True,
                alpha=200,
                min_action=0.,
                max_action=1.,
                n_substeps=15)

states = env.reset()
n_outputs = env.observation_space.shape[0]
print(states)
print(n_outputs)
print(env.tau, env.simulation_start_time, env.simulation_end_time)
print(env.action_space)
print(env.alpha)

# test substeps
observation, reward, done, _ = env.step([2])
substep_measurement_names, substep_measurement = env.get_substep_measurement()
print("current step is evenly divided into " +
      str(env.n_substeps) + " sub-steps!!!")
print(substep_measurement_names)
print(substep_measurement)
print(len(substep_measurement[0]))

# test cost and penalty
print("============================")
print("Cost at current step is "+str(env.get_cost()))
print("Maximum temperature violation at current step is " +
      str(env.get_temperature_violation()))

# test historical states
print("===========================")
states = env.reset()
print("t=0, Historical measurement is: ", env.history)
print("t=0, States are: ", states)
print()
observation, reward, done, _ = env.step([2])
print("t=1, Historical measurement is: ", env.history)
print("t=1, States are: ", observation)
print()
observation, reward, done, _ = env.step([3])
print("t=2, Historical measurement is: ", env.history)
print("t=2, States are: ", observation)
print("===========================\n")
states = env.reset()
print("t=0 after reset, Historical measurement is: ", env.history)
print("t=0 after reset, States are: ", states)

print("\nJModelicaCSSingleZoneEnv-price-v2 is successfully installed!!")
