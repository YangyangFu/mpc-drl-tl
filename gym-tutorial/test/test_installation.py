import gym
import gym_cart_jmodelica

env = gym.make("JModelicaCSCartPoleEnv-v0",
               m_cart=10,
               m_pole=1,
               theta_0=0.15,
               theta_dot_0=0,
               time_step=0.05,
               positive_reward=1,
               negative_reward=-100,
               force=15,
               log_level=2)
