from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division
import sys, os
# this is needed for simulating fmu in pythong 3 using pyfmi:/opt/conda/lib/python3.8/site-packages
PYFMI_PY3_CONDA_PATH = os.getenv("PYFMI_PY3_CONDA_PATH")
sys.path.insert(0, PYFMI_PY3_CONDA_PATH)

import numpy as np
import argparse
import json
import gym
# import gym_singlezone_jmodelica
import tianshou as ts
print(ts.__version__)
# import torch
from tianshou.data import ReplayBuffer, Batch
import pickle

time_step = 15*60.0
num_of_days = 7  # 31
max_number_of_steps = int(num_of_days*24*60*60.0 / time_step)

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='JModelicaCSSingleZoneEnv-action-v2')
parser.add_argument('--time-step', type=float, default=time_step)
parser.add_argument('--step-per-epoch', type=int, default=max_number_of_steps)
parser.add_argument('--seed', type=int, default=0)

# tunable parameters
parser.add_argument('--weight-energy', type=float, default= 100.)   
parser.add_argument('--weight-temp', type=float, default= 1.)  
parser.add_argument('--weight-action', type=float, default=10.) 
parser.add_argument('--buffer-size', type=int, default=4096) # equal to the step-per-epoch
parser.add_argument('--save-buffer-name', type=str, default='expert_MPC_JModelicaCSSingleZoneEnv-action-v2.pkl')
args = parser.parse_args()

def run_mpc_action_v0(args):

    def reverse_normalize_actions(normalized_action, min_action, max_action):
        normalized_action = np.array(normalized_action).reshape(-1,)
        min_action = np.array(min_action).reshape(-1,)
        max_action = np.array(max_action).reshape(-1,)

        # Reverse mapping from [0, 1] to [min_action, max_action]
        original_action = [normalized_action[i]*(max_action[i] - min_action[i]) + min_action[i] for i in range(len(normalized_action))]

        return original_action

    def make_building_env(args):
        import gym_singlezone_jmodelica
        
        weather_file_path = "USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw"
        mass_flow_nor = [0.55]
        n_next_steps = 4
        n_prev_steps = 4
        simulation_start_time = 201*24*3600.0
        simulation_end_time = simulation_start_time + args.step_per_epoch*args.time_step
        log_level = 0
        alpha = 1
        weight_energy = args.weight_energy #5.e4
        weight_temp = args.weight_temp #500.
        weight_action = args.weight_action

        def rw_func(cost, penalty, delta_action):
            if ( not hasattr(rw_func,'x')  ):
                rw_func.x = 0
                rw_func.y = 0

            cost = cost[0]
            penalty = penalty[0]
            delta_action = delta_action[0]
            if rw_func.x > cost:
                rw_func.x = cost
            if rw_func.y > penalty:
                rw_func.y = penalty

            #print("rw_func-cost-min=", rw_func.x, ". penalty-min=", rw_func.y)
            #res = (penalty * 500.0 + cost*5e4)/1000.0#!!!!!!!!!!!!!!!!!!
            res = -penalty*penalty * weight_temp + cost*weight_energy - delta_action*delta_action*weight_action
            
            return res

        env = gym.make(args.task,
                    mass_flow_nor = mass_flow_nor,
                    weather_file = weather_file_path,
                    n_next_steps = n_next_steps,
                    simulation_start_time = simulation_start_time,
                    simulation_end_time = simulation_end_time,
                    time_step = args.time_step,
                    log_level = log_level,
                    alpha = alpha,
                    rf = rw_func,
                    n_prev_steps=n_prev_steps)
        return env

    env = make_building_env(args)
    print("=== Environment is created ===")

    # read optimal control inputs
    with open('./mpc/R2/PH=48/u_opt.json') as f:
        opt = json.load(f)

    t_opt = opt['t_opt']
    u_opt = opt['u_opt']

    print(len(t_opt))
    print(len(u_opt))

    # initialize replay buffer
    expert_buffer = ReplayBuffer(max_number_of_steps)
    print("=== Replay buffer is created ===")

    # Run environment with MPC control inputs
    observation_prev = None

    # initial state
    init_states = env.reset()
    done = False

    for i in range(len(u_opt)):

        # Read optimal control input
        u = u_opt[i]
        u = reverse_normalize_actions(u, env.action_space.low, env.action_space.high)
        print("Action at step " + str(i) + " is " + str(u))

        # Step environment
        observation, reward, done, _ = env.step(u)

        # Determine terminated and truncated values
        terminated = 1 if done and i == len(u_opt) - 1 else 0
        truncated = 0  # Set truncated to 0 for all transitions
        
        # Save state transition into replay buffer
        if i == 0:
            expert_buffer.add(
                Batch(
                    obs=init_states, 
                    act=u, 
                    rew=reward, 
                    done=done, 
                    obs_next=observation,
                    terminated=terminated,
                    truncated=truncated))
        else:
            expert_buffer.add(
                Batch(
                    obs=observation_prev, 
                    act=u, 
                    rew=reward, 
                    done=done, 
                    obs_next=observation,
                    terminated=terminated,
                    truncated=truncated))

        if done:
            break  # Consider breaking if the episode ends

        observation_prev = observation

    print("=== Replay buffer is filled ===")

    # Save the buffer data as a pickle file
    with open(args.save_buffer_name, "wb") as f:
        # pickle.dump(transition_data, f)
        pickle.dump(expert_buffer, f)
    print(f"Data collected and saved to {args.save_buffer_name}")

if __name__ == '__main__':
    run_mpc_action_v0(args)