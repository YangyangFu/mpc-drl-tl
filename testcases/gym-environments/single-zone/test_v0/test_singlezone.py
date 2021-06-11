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

def train_qlearning(singlezone_env, max_number_of_steps=500, n_episodes=4, visualize=False):
    """
    Runs one experiment of Q-learning training on cart pole environment
    :param singlezone_env: environment RL agent will learn on.
    :param max_number_of_steps: maximum episode length.
    :param n_episodes: number of episodes to perform.
    :param visualize: flag if experiments should be rendered.
    :return: trained Q-learning agent, array of actual episodes length, execution time in s
    """

    start = time.time()
    n_outputs = singlezone_env.observation_space.shape[0]

    episode_lengths = np.array([])

    #bins of observation space - state space
    ts = singlezone_env.simulation_start_time
    te = ts + max_number_of_steps*singlezone_env.tau

    TZon_bins = _get_bins(273.15+10, 273.15+30, 4)
    TOut_bins = _get_bins(273.15+0, 273.15+40, 4)
    solar_bins = _get_bins(0, 2000, 4)
    power_bins = _get_bins(0, 10000, 4)
    
    learner = QLearner(n_states=4**n_outputs,
                       n_actions=singlezone_env.action_space.n,
                       learning_rate=0.2,
                       discount_factor=1,
                       exploration_rate=0.5,
                       exploration_decay_rate=0.99)

    for episode in range(n_episodes):
        observes = singlezone_env.reset()
        print(observes)
        t,TZon, TOut, solRad, powTot, TOut_pred1, TOut_pred2, TOut_pred3, solRad_pred1, solRad_pred2, solRad_pred3 = observes
        state = _get_state_index([_to_bin(TZon, TZon_bins),
                                    _to_bin(TOut, TOut_bins),
                                    _to_bin(solRad, solar_bins),
                                    _to_bin(powTot, power_bins),
                                    _to_bin(TOut_pred1, TOut_bins),
                                    _to_bin(TOut_pred2, TOut_bins),
                                    _to_bin(TOut_pred3, TOut_bins),
                                    _to_bin(solRad_pred1, solar_bins),
                                    _to_bin(solRad_pred2, solar_bins),   
                                    _to_bin(solRad_pred3, solar_bins)])
        print([_to_bin(TZon, TZon_bins),
                                    _to_bin(TOut, TOut_bins),
                                    _to_bin(solRad, solar_bins),
                                    _to_bin(powTot, power_bins),
                                    _to_bin(TOut_pred1, TOut_bins),
                                    _to_bin(TOut_pred2, TOut_bins),
                                    _to_bin(TOut_pred3, TOut_bins),
                                    _to_bin(solRad_pred1, solar_bins),
                                    _to_bin(solRad_pred2, solar_bins),   
                                    _to_bin(solRad_pred3, solar_bins)])
    
        print(state)

        action = learner.set_initial_state(state)

        for step in range(max_number_of_steps):
            if visualize:
                singlezone_env.render()

            observation, reward, done, _ = singlezone_env.step(action)

            t,TZon, TOut, solRad, powTot, TOut_pred1, TOut_pred2, TOut_pred3, solRad_pred1, solRad_pred2, solRad_pred3 = observation
            state_prime = _get_state_index([_to_bin(TZon, TZon_bins),
                                    _to_bin(TOut, TOut_bins),
                                    _to_bin(solRad, solar_bins),
                                    _to_bin(powTot, power_bins),
                                    _to_bin(TOut_pred1, TOut_bins),
                                    _to_bin(TOut_pred2, TOut_bins),
                                    _to_bin(TOut_pred3, TOut_bins),
                                    _to_bin(solRad_pred1, solar_bins),
                                    _to_bin(solRad_pred2, solar_bins),   
                                    _to_bin(solRad_pred3, solar_bins)])

            rew = reward[0][0]+reward[1][0]
            action = learner.learn_observation(state_prime, rew)
            print (action)
            if done or step == max_number_of_steps - 1:
                episode_lengths = np.append(episode_lengths, int(step + 1))
                break
    end = time.time()
    execution_time = end - start
    return learner, episode_lengths, execution_time


# Internal logic for state discretization
def _get_bins(lower_bound, upper_bound, n_bins):
    """
    Given bounds for environment state variable splits it into n_bins number of bins,
    taking into account possible values outside the bounds.

    :param lower_bound: lower bound for variable describing state space
    :param upper_bound: upper bound for variable describing state space
    :param n_bins: number of bins to receive
    :return: n_bins-1 values separating bins. I.e. the most left bin is opened from the left,
    the most right bin is open from the right.
    """
    return np.linspace(lower_bound, upper_bound, n_bins + 1)[1:-1]


def _to_bin(value, bins):
    """
    Transforms actual state variable value into discretized one,
    by choosing the bin in variable space, where it belongs to.

    :param value: variable value
    :param bins: sequence of values separating variable space
    :return: number of bin variable belongs to. If it is smaller than lower_bound - 0.
    If it is bigger than the upper bound
    """
    return np.digitize(x=[value], bins=bins)[0]


def _get_state_index(state_bins):
    """
    Transforms discretized environment state (represented as sequence of bin indexes) into an integer value.
    Value is composed by concatenating string representations of a state_bins.
    Received string is a valid integer, so it is converted to int.

    :param state_bins: sequence of integers that represents discretized environment state.
    Each integer is an index of bin, where corresponding variable belongs.
    :return: integer value corresponding to the environment state
    """

    #state = int("".join(map(lambda state_bin: str(state_bin), state_bins)))
    state = 0

    for i, state_bin in zip(range(len(state_bins)),state_bins):
        state += state_bin*4**i
    return state

def run_ql_experiments(n_experiments=1,
                       n_episodes=10,
                       visualize=False,
                        mass_flow_nor=[0.75],
                        weather_file='USA_CA_Riverside.Muni.AP.722869_TMY3.epw',
                        npre_step=3,
                        simulation_start_time=212*3600*24,
                        time_step=15*60.,
                        log_level=7):

#   from gym.envs.registration import register
    import gym_singlezone_jmodelica
    env_name = "JModelicaCSSingleZoneEnv-v0"

    trained_agent_s =[]
    episodes_length_s = []
    exec_time_s = []
    # env = gym.make(env, some_kwarg=your_vars)
    # need find a way to update the configuration at one time
    env = gym.make(env_name,
                mass_flow_nor=mass_flow_nor,
                weather_file=weather_file,
                npre_step=npre_step,
                simulation_start_time=simulation_start_time,
                time_step=time_step,
                log_level=log_level,
                alpha=0.01,
                nActions=101)
        
    for i in range(n_experiments):
        trained_agent, episodes_length, exec_time = train_qlearning(env,
                                                                    n_episodes=n_episodes,
                                                                    visualize=visualize)
        trained_agent_s.append(trained_agent)
        episodes_length_s.append(episodes_length)
        exec_time_s.append(exec_time)
        env.reset()

    env.close()
    # delete registered environment to avoid errors in future runs.
    # del gym.envs.registry.env_specs[env_name]
    return trained_agent_s, episodes_length_s, exec_time_s


def run_experiment_with_result_files(folder,
                        n_experiments=1,
                        n_episodes=10,
                        visualize=False,
                        mass_flow_nor=[0.75],
                        weather_file='USA_CA_Riverside.Muni.AP.722869_TMY3.epw',
                        npre_step=3,
                        simulation_start_time=212*3600*24,
                        time_step=15*60.,
                        log_level=7):
    """
    Runs experiments with the given configuration and writes episodes length of all experiment as one file
    and execution times of experiments as another.
    File names are composed from numerical experiment parameters
    in the same order as in function definition.
    Episodes length are written as 2d-array of shape (n_episodes, n_experiments):
    i-th row - i-th episode, j-th column - j-th experiment.

    Execution times are written as 1d-array of shape (n_experiments, ): j-th element - j-th experiment


    :param folder: folder for experiment result files
    :return: None
    """
    experiment_file_name_prefix = "{}/experiment_{}_{}".format(
        folder,
        n_experiments,
        n_episodes
    )
    _, episodes_lengths, exec_times = run_ql_experiments(n_experiments=n_experiments,
                                                         n_episodes=n_episodes,
                                                         visualize=visualize,
                                                        mass_flow_nor=mass_flow_nor,
                                                        weather_file=weather_file,
                                                        npre_step=npre_step,
                                                        simulation_start_time=simulation_start_time,
                                                        time_step=time_step,
                                                        log_level=log_level)
    print(episodes_lengths)
    n =experiment_file_name_prefix + "episodes_lengths.csv"
    X = np.transpose(episodes_lengths)
    print(X.shape)
    
    np.savetxt(fname=experiment_file_name_prefix + "episodes_lengths.csv",
               X=np.transpose(episodes_lengths),
               delimiter=",",
               fmt='%10.4f')
    np.savetxt(fname=experiment_file_name_prefix + "exec_times.csv",
               X=np.array(exec_times),
               delimiter=",",
               fmt='%10.4f')

if __name__ == "__main__":
    import time
    import os

    start = time.time()
    folder = "experiments_results"
    
    if not os.path.exists(folder):
        os.mkdir(folder)

    # following experiments rake significant amount of time, so it is advised to run only one of them at once
    run_experiment_with_result_files(folder,
                                        n_experiments=1,
                                        n_episodes=1,
                                        visualize=False,
                                        mass_flow_nor=[0.75],
                                        weather_file='USA_CA_Riverside.Muni.AP.722869_TMY3.epw',
                                        npre_step=3,
                                        simulation_start_time=212*3600*24.,
                                        time_step=15*60.,
                                        log_level=7)

    end = time.time()
    print("Total execution time {:.2f} seconds".format(end-start))
