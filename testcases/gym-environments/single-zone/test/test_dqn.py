# -*- coding: utf-8 -*-
'''This example is 
'''
from __future__ import print_function
from __future__ import absolute_import, division

import logging
from q_learning import QLearner
import gym
import numpy as np
import math
import time
import os

import gym_singlezone_jmodelica
from evaluate_tools.paint import plot_one_ep


import sys
sys.path.append('./DQN/')
from memory import Memory
from neural import NeuralNet
from agent import Agent




def raw_agent(path):
    explore_high = 1.0
    explore_low = 0.1
    num_ctr_step = 2976
    explore_decay = (explore_high-explore_low)/(num_ctr_step*100.0)
    agent = Agent(layout=[11,50,100,200,400,5],
            batch=48,
            explore=explore_high,
            explore_l=explore_low,
            explore_d=explore_decay,
            learning=0.0003,
            decay=0.99,
            k=48*5,
            m_size=48*31,
            pa=path)
    return agent
    
    
def regul_state(state):
    state1 = state
    state1[0] = (int(state1[0]) % (3600*24)) * 1.0 / 3600.0
    state1[1] = (state1[1] - 10) / 30.0
    state1[2] = (state1[2] - 10) / 30.0
    state1[3] = state1[3] / 100.0
    state1[4] = state1[4]
    
    state1[5] = (state1[5] - 10) / 30.0
    state1[6] = (state1[6] - 10) / 30.0
    state1[7] = (state1[7] - 10) / 30.0
    state1[8] /= 100.0
    state1[9] /= 100.0
    state1[10] /= 100.0
    return state1

def model_simulation(folder, path):
    
    env_name = "JModelicaCSSingleZoneEnv-v0"
    weather_file_path = "./USA_CA_Riverside.Muni.AP.722869_TMY3.epw"
    time_step = 15*60.0
    mass_flow_nor = [0.75]
    npre_step = 3
    simulation_start_time = 212*24*3600.0
    log_level = 7
    num_eps = 1
    
    env = gym.make(env_name,
                   mass_flow_nor = mass_flow_nor,
                   weather_file = weather_file_path,
                   npre_step = npre_step,
                   simulation_start_time = simulation_start_time,
                   time_step = time_step,
                   log_level = log_level)
                 
    max_number_of_steps = int(31*24*60*60.0 / time_step)
    #n_outputs = env.observation_space.shape[0]
    
    agent = raw_agent(path)
    agent.initialize(path)
    print('DRL agent created!')
    
    history_Z_T = [[]]
    history_Env_T = [[]]
    history_Action = [[]]
    
    for ep in range(num_eps):
        observation = env.reset()
        cur_time = env.start
        Z_T, Env_T, Solar_R, power, Env_T1, Env_T2, Env_T3, Solar_R1, Solar_R2, Solar_R3 = observation
        Z_T -= 273.15
        Env_T -= 273.15
        Env_T1 -= 273.15
        Env_T2 -= 273.15
        Env_T3 -= 273.15
        state = [cur_time, Z_T, Env_T, Solar_R, power, Env_T1, Env_T2, Env_T3, Solar_R1, Solar_R2, Solar_R3]
        state = regul_state(state)
        
        for step in range(max_number_of_steps):
            action = agent.policy(state)
            
            observation, reward, done, _ = env.step(action)
            #cur_time, 
            cur_time = env.start
            Z_T, Env_T, Solar_R, power, Env_T1, Env_T2, Env_T3, Solar_R1, Solar_R2, Solar_R3 = observation
            Z_T -= 273.15
            Env_T -= 273.15
            Env_T1 -= 273.15
            Env_T2 -= 273.15
            Env_T3 -= 273.15
            
            state_prime = [cur_time, Z_T, Env_T, Solar_R, power, Env_T1, Env_T2, Env_T3, Solar_R1, Solar_R2, Solar_R3]
            state_prime = regul_state(state_prime)
            
            agent.learning(state,action,reward,state_prime)
            
            history_Z_T[ep].append([Z_T])
            history_Env_T[ep].append(Env_T)
            history_Action[ep].append([action])
            
            if done or step == max_number_of_steps - 1:
                break
            state = state_prime
            
    np.save('./'+folder+'/history_Z_T.npy', history_Z_T)
    np.save('./'+folder+'/history_Env_T.npy', history_Env_T)
    np.save('./'+folder+'/history_Action.npy', history_Action)


if __name__ == "__main__":
    
    start = time.time()
    folder = "dqn_experiments_results"
    if not os.path.exists(folder):
        os.mkdir(folder)
    model_simulation(folder, './'+folder)
    end = time.time()
    print("Total execution time {:.2f} seconds".format(end-start))
    
    
    history_Z_T = np.load("./"+folder+"/history_Z_T.npy")
    history_Env_T = np.load("./"+folder+"/history_Env_T.npy")
    plot_one_ep(num_zone = 1, history_Z_T = history_Z_T, history_Env_T = history_Env_T, ep = 1, fig_path_name = "./"+folder+"/ON_OFF_simulation.png")
    