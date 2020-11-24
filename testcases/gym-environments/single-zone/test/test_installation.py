import gym
import gym_singlezone_jmodelica

env = gym.make("JModelicaCSSingleZoneEnv-v0",
                mass_flow_nor=0.75,
                weather_file='USA_CA_Riverside.Muni.AP.722869_TMY3.epw',
                npre_step=3,
                simulation_start_time=3600*24,
                time_step=15*60.,
                log_level=7)
states = env.reset()
print(states)
