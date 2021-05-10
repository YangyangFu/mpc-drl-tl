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

import gym_singlezone_temperature
from evaluate_tools.paint import plot_one_ep, getViolation, plot_one_action_ep


import sys
sys.path.append('./DQN/')
from memory import Memory
from neural import NeuralNet
from agent import Agent


training_epochs = 50

def raw_agent(path):
    explore_high = 1.0
    explore_low = 0.1
    num_ctr_step = 24*7 #2976 # one month
    explore_decay = (explore_high-explore_low)/(num_ctr_step*training_epochs*1.0)
    agent = Agent(layout=[11,50,100,200,400,37],
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
    tim_env = 0.0
    tim_ctl = 0.0
    tim_learn = 0.0

    env_name = "JModelicaCSSingleZoneTemperatureEnv-v0"
    weather_file_path = "./USA_CA_Riverside.Muni.AP.722869_TMY3.epw"
    time_step = 15*60.0
    mass_flow_nor = [0.75]
    npre_step = 3
    simulation_start_time = 212*24*3600.0
    log_level = 7
    num_eps = training_epochs
    
    env = gym.make(env_name,
                   mass_flow_nor = mass_flow_nor,
                   weather_file = weather_file_path,
                   npre_step = npre_step,
                   simulation_start_time = simulation_start_time,
                   time_step = time_step,
                   log_level = log_level)
                 
    num_of_days = 7#31
    max_number_of_steps = int(num_of_days*24*60*60.0 / time_step)
    #n_outputs = env.observation_space.shape[0]
    
    agent = raw_agent(path)
    agent.initialize(path)
    print('DRL agent created!')
    
    history_Z_T = [[] for i in range(num_eps)]
    history_Env_T = [[] for i in range(num_eps)]
    history_Action = [[] for i in range(num_eps)]
    history_Reward = [[] for i in range(num_eps)]
    
    for ep in range(num_eps):
        observation = env.reset()
        #cur_time = env.start
        cur_time, Z_T, Env_T, Solar_R, power, Env_T1, Env_T2, Env_T3, Solar_R1, Solar_R2, Solar_R3 = observation
        Z_T -= 273.15
        Env_T -= 273.15
        Env_T1 -= 273.15
        Env_T2 -= 273.15
        Env_T3 -= 273.15
        state = [cur_time, Z_T, Env_T, Solar_R, power, Env_T1, Env_T2, Env_T3, Solar_R1, Solar_R2, Solar_R3]
        state = regul_state(state)
        
        for step in range(max_number_of_steps):
            tim_begin = time.time()
            action = agent.policy(state)
            tim_end = time.time()
            tim_ctl += tim_begin - tim_end
            print("Action is: "+str(action))

            tim_begin = time.time()
            observation, reward, done, _ = env.step(action)
            tim_end = time.time()
            tim_env += tim_begin - tim_end

            #cur_time = env.start
            cur_time, Z_T, Env_T, Solar_R, power, Env_T1, Env_T2, Env_T3, Solar_R1, Solar_R2, Solar_R3 = observation
            Z_T -= 273.15
            Env_T -= 273.15
            Env_T1 -= 273.15
            Env_T2 -= 273.15
            Env_T3 -= 273.15
            
            state_prime = [cur_time, Z_T, Env_T, Solar_R, power, Env_T1, Env_T2, Env_T3, Solar_R1, Solar_R2, Solar_R3]
            state_prime = regul_state(state_prime)
            
            tim_begin = time.time()
            agent.learning(state,action,reward,state_prime)
            tim_end = time.time()
            tim_learn += tim_begin - tim_end
            
            history_Z_T[ep].append([Z_T])
            history_Env_T[ep].append(Env_T)
            history_Action[ep].append([action])
            history_Reward[ep].append([reward])
            
            if done or step == max_number_of_steps - 1:
                break
            state = state_prime
            
    np.save('./'+folder+'/history_Z_T.npy', history_Z_T)
    np.save('./'+folder+'/history_Env_T.npy', history_Env_T)
    np.save('./'+folder+'/history_Action.npy', history_Action)
    np.save('./'+folder+'/history_Reward.npy', history_Reward)
    return tim_env, tim_learn, tim_ctl


if __name__ == "__main__":
    
    start = time.time()
    folder = "dqn_experiments_results"
    if not os.path.exists(folder):
        os.mkdir(folder)
    tim_env, tim_learn, tim_ctl = model_simulation(folder, './'+folder)
    end = time.time()
    print("tim_env, tim_learn, tim_ctl = ", -tim_env, -tim_learn, -tim_ctl)
    print("Total execution time {:.2f} seconds".format(end-start))
    
    
    history_Z_T = np.load("./"+folder+"/history_Z_T.npy")
    history_Env_T = np.load("./"+folder+"/history_Env_T.npy")
    history_Action = np.load("./"+folder+"/history_Action.npy")

    #plot_one_action_ep(num_zone = 1, history_Z_T = history_Z_T, history_Env_T = history_Env_T, history_action = history_Action, ep = 1, fig_path_name = "./"+folder+"/DQN_simulation.png")
    plot_one_ep(num_zone = 1, history_Z_T = history_Z_T, history_Env_T = history_Env_T, ep = 1, fig_path_name = "./"+folder+"/DQN_simulation.png")
    getViolation(num_zone = 1, ep = 1, history_Z_T = history_Z_T, delCtrl=15*60.0, num_days = 7)
    

    #history_Reward = np.load("./"+folder+"/history_Reward.npy")
    #print(history_Reward[2][912][0])