"""
Calculation of MPC/RBC rewards
1. MPC/RBC rewards
"""
from __future__ import print_function
from __future__ import absolute_import, division

import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import json
# load testbed
from pyfmi import load_fmu

#==========================================================
##              General Settings
# =======================================================
# simulation setup
ts = 201*24*3600.#+13*24*3600
nday = 7
period = nday*24*3600.
te = ts + period
dt = 15*60.
nsteps_h = int(3600//dt)

# define some filters to save simulation time using fmu
measurement_names = ['time','TRoo','TOut','PTot','fcu.uFan']

##########################################
##          Baseline Simulation
## =========================================
# DEFINE MODEL
# ------------
baseline = load_fmu('SingleZoneFCUBaseline.fmu')

## fmu settings
options = baseline.simulate_options()
options['ncp'] = 672
options['initialize'] = True
options['result_handling'] = 'memory'
options['filter'] = measurement_names

## construct optimal input for fmu
baseline.set("zon.roo.T_start", 273.15+25)
res_base = baseline.simulate(start_time = ts,
                    final_time = te, 
                    options = options)

################################################
##           MPC Final Simulation
## =============================================

def simulate_mpc(ts, te, dt, nday, u_opt_path, weights, measurement_names):
    # read optimal control inputs
    with open(u_opt_path+'/u_opt.json') as f:
        opt = json.load(f)

    t_opt = opt['t_opt']
    u_opt = opt['u_opt']

    print(len(t_opt))
    print(len(u_opt))

    ### 1- Load virtual building model
    hvac = load_fmu('SingleZoneFCU.fmu')

    ## fmu settings
    options = hvac.simulate_options()
    options['ncp'] = 15
    options['initialize'] = True
    options['result_handling'] = 'memory'
    options['filter'] = measurement_names
    res_mpc = []
    hvac.set("zon.roo.T_start", 273.15+25)
    # main loop - do step
    t = ts
    i = 0
    while t < te:
        u = u_opt[i]
        hvac.set("uFan", u)
        ires = hvac.simulate(start_time = t,
                    final_time = t+dt, 
                    options = options)
        res_mpc.append(ires)

        t += dt 
        i += 1
        options['initialize'] = False

    ################################################################
    ##           Compare MPC/DRL with Baseline
    ## =============================================================

    # read measurements
    measurement_mpc = {}
    measurement_base = {}
    measurement_ddqn = {}

    for name in measurement_names:
        measurement_base[name] = res_base[name]
        # get mpc results
        value_name_mpc=[]
        for ires in res_mpc:
            value_name_mpc += list(ires[name])
        measurement_mpc[name] = np.array(value_name_mpc)

    ## simulate baseline
    occ_start = 8
    occ_end = 18

    price_tou = [0.02987, 0.02987, 0.02987, 0.02987,
                0.02987, 0.02987, 0.04667, 0.04667,
                0.04667, 0.04667, 0.04667, 0.04667,
                0.15877, 0.15877, 0.15877, 0.15877,
                0.15877, 0.15877, 0.15877, 0.04667,
                0.04667, 0.04667, 0.02987, 0.02987]*nday

    # save baseline and mpc measurements from simulation
    ## save interpolated measurement data for comparison
    def interpolate_dataframe(df,new_index):
        """Interpolate a dataframe along its index based on a new index
        """
        df_out = pd.DataFrame(index=new_index)
        df_out.index.name = df.index.name

        for col_name, col in df.items():
            df_out[col_name] = np.interp(new_index, df.index, col)    
        return df_out

    measurement_base = pd.DataFrame(measurement_base,index=measurement_base['time'])
    measurement_mpc = pd.DataFrame(measurement_mpc,index=measurement_mpc['time'])

    tim_intp = np.arange(ts,te+1,dt)
    measurement_base = interpolate_dataframe(measurement_base[['PTot','TRoo', 'fcu.uFan']],tim_intp)
    measurement_mpc = interpolate_dataframe(measurement_mpc[['PTot','TRoo', 'fcu.uFan']],tim_intp)

    #measurement_base = measurement_base.groupby(measurement_base.index//900).mean()
    #measurement_mpc = measurement_mpc.groupby(measurement_mpc.index//900).mean()

    def rw_func(cost, penalty, delta_action):

        res = - weights[0]*cost - weights[1]*penalty*penalty \
            - delta_action*delta_action*weights[2]

        return res

    def get_rewards(u, Ptot, TZone, price_tou):
        n = len(u)
        energy = []
        energy_cost = []
        penalty = []
        delta_action = []
        rewards = []

        u_prev = [0.] + list(u)
        u_prev = u_prev[:-1]

        for i in range(n):
            # assume 1 step is 15 minutes and data starts from hour 0
            hindex = (i % (nsteps_h*24))//nsteps_h
            power = Ptot[i]
            price = price_tou[hindex]
            # the power should divide by 1000
            energy.append(power/1000./nsteps_h)
            energy_cost.append(power/1000./nsteps_h*price)

            # zone temperature penalty
            number_zone = 1

            # zone temperature bounds - need check with the high-fidelty model
            T_upper = np.array([30.0 for j in range(24)])
            T_upper[occ_start:occ_end] = 26.0
            T_lower = np.array([12.0 for j in range(24)])
            T_lower[occ_start:occ_end] = 22.0

            overshoot = []
            undershoot = []
            for k in range(number_zone):
                overshoot.append(
                    np.array([float((TZone[i] - 273.15) - T_upper[hindex]), 0.0]).max())
                undershoot.append(
                    np.array([float(T_lower[hindex] - (TZone[i]-273.15)), 0.0]).max())

            penalty.append(sum(np.array(overshoot)) +
                           sum(np.array(undershoot)))

            # action changes
            delta_action.append(abs(u[i] - u_prev[i]))

            # sum up for rewards
            rewards.append(
                rw_func(energy_cost[-1], penalty[-1], delta_action[-1]))

        return np.array([energy, energy_cost, penalty, delta_action, rewards]).transpose()

    #### get rewards
    #================================================================================
    rewards_base = get_rewards(measurement_base['fcu.uFan'].values, measurement_base['PTot'].values,measurement_base['TRoo'].values,price_tou)
    rewards_mpc = get_rewards(measurement_mpc['fcu.uFan'].values, measurement_mpc['PTot'].values,measurement_mpc['TRoo'].values,price_tou)

    rewards_base = pd.DataFrame(rewards_base,columns=[['energy','ene_cost','penalty', 'delta_action', 'rewards']])
    rewards_mpc = pd.DataFrame(rewards_mpc,columns=[['energy','ene_cost','penalty','delta_action', 'rewards']])

    # get rewards - DRL - we can either read from training results or 
    # recalculate using the method for mpc and baseline (very time consuming for multi-epoch training)
    ### ===============
    mpc_rbc_rewards = {'base': {'rewards': float(rewards_base['rewards'].sum()),
                                'energy': float(rewards_base['energy'].sum()),
                                'ene_cost': float(rewards_base['ene_cost'].sum()),
                                'total_temp_violation': float(rewards_base['penalty'].sum()/4),
                                'max_temp_violation': float(rewards_base['penalty'].max()),
                                'delta_action': float(rewards_base['delta_action'].sum()/4),
                                'temp_violation_squared': float((rewards_base['penalty']**2).sum()),
                                'delta_action_sqaured': float((rewards_base['delta_action']**2).sum())},
                       'mpc': {'rewards': float(rewards_mpc['rewards'].sum()),
                               'energy': float(rewards_mpc['energy'].sum()),
                               'ene_cost': float(rewards_mpc['ene_cost'].sum()),
                               'total_temp_violation': float(rewards_mpc['penalty'].sum()/4),
                               'max_temp_violation': float(rewards_mpc['penalty'].max()),
                               'delta_action': float(rewards_mpc['delta_action'].sum()/4),
                               'temp_violation_squared': float((rewards_mpc['penalty']**2).sum()),
                               'delta_action_sqaured': float((rewards_mpc['delta_action']**2).sum())}
                       }
    with open(u_opt_path+'/mpc_rewards.json', 'w') as outfile:
        json.dump(mpc_rbc_rewards, outfile)


# Which MPC rewards
weights = [100., 1., 10.]
PHs=[4, 8 ,16, 32, 48, 96]
for PH in PHs:
    u_opt_path = './mpc/R2/PH='+str(PH)
    simulate_mpc(ts, te, dt, nday, u_opt_path, weights, measurement_names)
