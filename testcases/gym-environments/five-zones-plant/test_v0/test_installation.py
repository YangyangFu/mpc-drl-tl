from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division
import numpy as np
import gym
import gym_fivezoneplant_jmodelica
import tianshou as ts
## ====================================================
#    Testing JModelicaCSFiveZonePlantEnv-v0 installation
## ===================================================
simulation_start_time=204*3600*24.
simulation_end_time=204*3600*24.+3600*12.
time_step=15*60.

env = gym.make("JModelicaCSFiveZonePlantEnv-v0",
                weather_file='USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw',
                npre_step=5,
                simulation_start_time=simulation_start_time,
                simulation_end_time=simulation_end_time,
                time_step=time_step,
                log_level=7,
                fmu_result_handling='memory',
                fmu_result_ncp=100,
                filter_flag=True,
                alpha=200,
                min_action=np.array([-1., -1., -1.]),
                max_action=np.array([1., 1., 1.]),
                n_substeps=15)

states = env.reset()
n_outputs = env.observation_space.shape[0]
print("states:")
print(states)
print("n_outputs:")
print(n_outputs)
print("(tau, simulation_start_time, simulation_end_time)")
print(env.tau, env.simulation_start_time, env.simulation_end_time)
print("action_space")
print(env.action_space)
print("alpha:")
print(env.alpha)
print(env.min_action, env.max_action)

# test substeps
max_number_of_steps=50
for step in range(max_number_of_steps):
    observation, reward, done, _ = env.step([-0.1+step*0.01,-0.5+step*0.01,0.3-step*0.01])
    if done or step == max_number_of_steps - 1:
        print("Final step:"+str(step))
        break

substep_measurement_names, substep_measurement=env.get_substep_measurement()
print("current step is evenly divided into "+str(env.n_substeps) + " sub-steps!!!")
print(substep_measurement_names)
print(substep_measurement)
print (len(substep_measurement[0]))

# test cost and penalty
print("============================")
print("Cost at current step is "+str(env.get_cost())) #[-3.3990643591232135] KWh in 15-min timestep
print("Maximum temperature violation at current step is "+str(env.get_temperature_violation())) #[-0.0, -0.031387453569379886, -0.0, -0.0, -0.0] K
print("\nJModelicaCSFiveZonePlantEnv-v0 is successfully installed!!" )