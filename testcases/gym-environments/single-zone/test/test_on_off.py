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


def ON_OFF_controller(T_lower, T_upper, T_cur):
    if T_cur > T_upper:
        return 4
    else:
        return 0

def model_simulation(folder):
    
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
                 
    max_number_of_steps = int(3*24*60*60.0 / time_step)
    #n_outputs = env.observation_space.shape[0]
    
    action = 0
    
    history_Z_T = [[]]
    history_Env_T = [[]]
    history_Action = [[]]
    
    for ep in range(num_eps):
        observation = env.reset()
        action = ON_OFF_controller(19.0, 24.0, observation[0]-273.15)
        
        for step in range(max_number_of_steps):
            observation, reward, done, _ = env.step(action)
            #cur_time, 
            cur_time = env.start
            Z_T, Env_T, Solar_R, power, Env_T1, Env_T2, Env_T3, Solar_R1, Solar_R2, Solar_R3 = observation
            Z_T -= 273.15
            Env_T -= 273.15
            Env_T1 -= 273.15
            Env_T2 -= 273.15
            Env_T3 -= 273.15
            action = ON_OFF_controller(19.0, 24.0, Z_T)
            
            history_Z_T[ep].append([Z_T])
            history_Env_T[ep].append(Env_T)
            history_Action[ep].append([action])
            
            
            
            if done or step == max_number_of_steps - 1:
                break
                
    np.save('./'+folder+'/history_Z_T.npy', history_Z_T)
    np.save('./'+folder+'/history_Env_T.npy', history_Env_T)
    np.save('./'+folder+'/history_Action.npy', history_Action)


if __name__ == "__main__":
    
    start = time.time()
    folder = "experiments_results"
    if not os.path.exists(folder):
        os.mkdir(folder)
    model_simulation(folder)
    end = time.time()
    print("Total execution time {:.2f} seconds".format(end-start))
    
    
    history_Z_T = np.load("./"+folder+"/history_Z_T.npy")
    history_Env_T = np.load("./"+folder+"/history_Env_T.npy")
    plot_one_ep(num_zone = 1, history_Z_T = history_Z_T, history_Env_T = history_Env_T, ep = 1, fig_path_name = "./"+folder+"/ON_OFF_simulation.png")
    