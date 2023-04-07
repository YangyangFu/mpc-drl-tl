from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division
import sys, os
# this is needed for simulating fmu in pythong 3 using pyfmi:/opt/conda/lib/python3.8/site-packages
PYFMI_PY3_CONDA_PATH = os.getenv("PYFMI_PY3_CONDA_PATH")
sys.path.insert(0, PYFMI_PY3_CONDA_PATH)
print(sys.path)
import gym
import env_example
#os.path.abspath(os.path.join(os.path.dirname('settings.py'),os.path.pardir))
# import tianshou as ts
# print(ts.__version__)

env = gym.make("Example-v1",
                )



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
step = env.step(0)
print('**************')
print(type(step))
print('State is {}  \nReward is {}\nTerminated is {}'.format(step[0],step[1],step[2]))

# # # test weather forecast
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
