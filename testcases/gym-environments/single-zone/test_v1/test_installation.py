import gym
import gym_singlezone_jmodelica
import tianshou as ts

env = gym.make("JModelicaCSSingleZoneEnv-v0",
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
                nActions=101)
states = env.reset()
n_outputs = env.observation_space.shape[0]
print(states)
print(env.tau, env.simulation_start_time)
print(env._get_action_space(),env.nActions)
print(env.alpha)
print(ts.__version__)

env = gym.make("JModelicaCSSingleZoneEnv-v1",
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
print(ts.__version__)