from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division
import sys, os
# this is needed for simulating fmu in pythong 3 using pyfmi:/opt/conda/lib/python3.8/site-packages
PYFMI_PY3_CONDA_PATH = os.getenv("PYFMI_PY3_CONDA_PATH")
sys.path.insert(0, PYFMI_PY3_CONDA_PATH)
print(sys.path)
import gym
import gym_singlezone_rom
import time

import random
import matplotlib.pyplot as plt
#os.path.abspath(os.path.join(os.path.dirname('settings.py'),os.path.pardir))
# import tianshou as ts
# print(ts.__version__)

env = gym.make("SingleZoneEnv-ANN-v1",
                mass_flow_nor=0.55,
                weather_file='USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw',
                n_next_steps=4,
                simulation_start_time=3600*24,
                simulation_end_time=3600*24*7,  # for the next week?
                time_step=15*60,
                log_level=7,
                alpha=200,
                nActions=11,
                # n_substeps=15,
                m_prev_steps=4)

# basic information
print('The interval in seconds is {}.\nThe start time is {}. \nThe end time is {}'.format(env.tau, env.simulation_start_time, env.simulation_end_time))
print(env.alpha)

#test action and obsevation space
action_space = env.action_space
print('The action space is {}'.format(action_space))
observation_space = env.observation_space
print('The shape of observation space is {}'.format(observation_space.shape))

reset = env.reset()
print('>>>>>>>>>>')
print(type(reset))
print('Reset state is {}'.format(reset[0]))

# test action changes
step = env.step(10)
print('**************')
print(type(step))
print('State is {}  \nReward is {}\nTerminated is {}'.format(step[0],step[1],step[2]))

simulation_start_time=3600*24
simulation_end_time=3600*24*2
i = simulation_start_time
rewards = []
while i<simulation_end_time:
    step = env.step(random.randint(0,10))
    print('**************')
    print('State is {}  \nReward is {}\nTerminated is {}'.format(step[0],step[1],step[2]))
    rewards.append(step[1])
    i += 900
fig, ax = plt.figure()
plt.plot(rewards)
plt.ylabel('reward')
plt.savefig('rewards.png', bbox_inches = 'tight', dpi = 300)

# # test weather forecast
# temp = env.predictor(4)
# # steps

# # observation, reward, done, _ = env.step(2)
# p =env.step(4)
# print('predicted power is: {}'.format(p))
# history_data = env.history_data
# substep_measurement_names, substep_measurement=env.get_substep_measurement()
# print("current step is evenly divided into "+str(env.n_substeps) + " sub-steps!!!")
# print(substep_measurement_names)
# print(substep_measurement)
# print (len(substep_measurement[0]))

# # test cost and penalty
# print("============================")
# print("Cost at current step is "+str(env.get_cost()))
# print("Maximum temperature violation at current step is "+str(env.get_temperature_violation()))
# print("Action change at current step is "+str(env.get_action_changes()))

# # test historical states
# print("===========================")
# states=env.reset()
# print("t=0, Historical measurement is: ", env.history)
# print("t=0, States are: ", states)
# print("t=0, Action change is "+str(env.get_action_changes()))
# print()
# observation, reward, done, _ = env.step(2)
# print("t=1, Historical measurement is: ", env.history)
# print("t=1, States are: ", observation)
# print("t=1, Action change is "+str(env.get_action_changes()))
# print()
# observation, reward, done, _ = env.step(3)
# print("t=2, Historical measurement is: ", env.history)
# print("t=2, States are: ", observation)
# print("t=2, Action change is "+str(env.get_action_changes()))
# print("===========================\n")
# states=env.reset()
# print("t=0 after reset, Historical measurement is: ", env.history)
# print("t=0 after reset, States are: ", states)
# print("t=0 after reset, Action change is "+str(env.get_action_changes()))

# print("\nJModelicaCSSingleZoneEnv-action-v1 is successfully installed!!")
