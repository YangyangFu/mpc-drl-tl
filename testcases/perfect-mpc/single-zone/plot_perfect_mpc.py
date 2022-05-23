from __future__ import print_function
from __future__ import absolute_import, division

from pyfmi import load_fmu
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import re
import os
import argparse
# load testbed

def find_all_files(root_dir):
    """Find all files under root_dir according to relative pattern."""
    sub_dirs=[]
    for it in os.scandir(root_dir):
        if it.is_dir():
            sub_dirs.append(it.path)

    file_list = []
    
    for sub_dir in sub_dirs:
        data_path = os.path.join(sub_dir, 'u_opt.json')
        file_list.append(data_path)

    return file_list

def simulate_baseline(ts, te, measurement_names):
    baseline = load_fmu('SingleZoneFCUBaseline.fmu')

    ## fmu settings
    options = baseline.simulate_options()
    options['ncp'] = 500
    options['initialize'] = True

    ## construct optimal input for fmu
    baseline.set("zon.roo.T_start", 273.15+25)
    res_base = baseline.simulate(start_time=ts,
                                final_time=te,
                                options=options)
    measurement_base = {}
    for name in measurement_names:
        measurement_base[name] = res_base[name]

    return measurement_base

def interpolate_dataframe(df, new_index):
    """Interpolate a dataframe along its index based on a new index
    """
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name

    for col_name, col in df.items():
        df_out[col_name] = np.interp(new_index, df.index, col)
    return df_out
    
def simulate_mpc(u_opt, ts, te, dt, measurement_names):
    # read optimal control inputs
    with open(u_opt) as f:
        opt = json.load(f)

    t_opt = opt['t_opt']
    u_opt = opt['u_opt']

    ### 1- Load virtual building model
    mpc = load_fmu('SingleZoneFCU.fmu')

    ## fmu settings
    options = mpc.simulate_options()
    options['ncp'] = 15
    options['initialize'] = True
    options['result_handling'] = 'memory'
    options['filter'] = measurement_names
    res_mpc = []

    # main loop - do step
    t = ts
    i = 0

    mpc.set("zon.roo.T_start", 273.15+25)
    while t < te:
        u = u_opt[i]
        mpc.set("uFan", u[0])
        ires = mpc.simulate(start_time=t,
                            final_time=t+dt,
                            options=options)
        res_mpc.append(ires)

        t += dt
        i += 1
        options['initialize'] = False
    measurement_mpc = {}
    for name in measurement_names:
        # get mpc results
        value_name_mpc = []
        for ires in res_mpc:
            value_name_mpc += list(ires[name])
        measurement_mpc[name] = np.array(value_name_mpc)

    return measurement_mpc

def plot(ts, te, dt, nday, PH, measurement_base, measurement_mpc, path):
    ## simulate baseline
    occ_start = 8
    occ_end = 18
    nsteps_h = int(3600./dt)

    tim = np.arange(ts, te, dt)
    T_upper = np.array([30.0 for i in tim])
    #T_upper[occ_start*4:(occ_end-1)*4] = 26.0
    T_lower = np.array([12.0 for i in tim])
    #T_lower[occ_start*4:(occ_end-1)*4] = 22.0
    for i in range(nday):
        T_upper[24*nsteps_h*i+occ_start*nsteps_h:24 *
                nsteps_h*i+occ_end*nsteps_h] = 26.0
        T_lower[24*nsteps_h*i+occ_start*nsteps_h:24 *
                nsteps_h*i+occ_end*nsteps_h] = 22.0

    price_tou = [0.02987, 0.02987, 0.02987, 0.02987,
                0.02987, 0.02987, 0.04667, 0.04667,
                0.04667, 0.04667, 0.04667, 0.04667,
                0.15877, 0.15877, 0.15877, 0.15877,
                0.15877, 0.15877, 0.15877, 0.04667,
                0.04667, 0.04667, 0.02987, 0.02987]*nday

    xticks = np.arange(ts, te+1, 12*3600)
    xticks_label = np.arange(0, 24*nday+1, 12)

    plt.figure(figsize=(16, 12))
    plt.subplot(411)
    price_plot = price_tou[:]
    price_plot.append(price_plot[0])
    plt.step(np.arange(ts, te+1, 3600.), price_plot, where='post')
    plt.grid(True)
    plt.xticks(xticks, [])
    plt.ylabel('Price ($/kW)')

    plt.subplot(412)
    plt.plot(measurement_base['time'],
            measurement_base['fcu.uFan'], 'b--', label='Baseline')
    plt.plot(measurement_mpc['time'],
            measurement_mpc['fcu.uFan'], 'r-', label='MPC')
    plt.grid(True)
    plt.xticks(xticks, [])
    plt.legend()
    plt.ylabel('Fan Speed')

    plt.subplot(413)
    plt.plot(measurement_base['time'],
            measurement_base['TRoo']-273.15, 'b--', label='Baseline')
    plt.plot(measurement_mpc['time'],
            measurement_mpc['TRoo']-273.15, 'r-', label='MPC')
    plt.plot(tim, T_upper, 'g-.', lw=1, label='Bounds')
    plt.plot(tim, T_lower, 'g-.', lw=1)
    plt.grid(True)
    plt.xticks(xticks, [])
    plt.legend()
    plt.ylabel('Room Temperature [C]')

    plt.subplot(414)
    plt.plot(measurement_base['time'],
            measurement_base['PTot'], 'b--', label='Baseline')
    plt.plot(measurement_mpc['time'], measurement_mpc['PTot'], 'r-', label='MPC')
    plt.grid(True)
    plt.xticks(xticks, xticks_label)
    plt.ylabel('Total [W]')
    plt.legend()
    plt.savefig(os.path.join(path, 'mpc-vs-rbc-'+str(PH)+'.pdf'))
    plt.savefig(os.path.join(path, 'mpc-vs-rbc-'+str(PH)+'.png'))


def calculate_metrics(ts, te, dt, measurement_base, measurement_mpc, weights, path):
    occ_start = 8
    occ_end = 18
    nsteps_h = int(3600//dt)

    price_tou = [0.02987, 0.02987, 0.02987, 0.02987,
                 0.02987, 0.02987, 0.04667, 0.04667,
                 0.04667, 0.04667, 0.04667, 0.04667,
                 0.15877, 0.15877, 0.15877, 0.15877,
                 0.15877, 0.15877, 0.15877, 0.04667,
                 0.04667, 0.04667, 0.02987, 0.02987]*nday

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

    def get_rewards(u, Ptot,TZone,price_tou):
        n = len(u)
        energy_cost = []
        penalty = []
        delta_action = []
        rewards = []

        u_prev = [0.] + list(u)
        u_prev = u_prev[:-1]

        for i in range(n):
            # assume 1 step is 15 minutes and data starts from hour 0
            hindex = (i%(nsteps_h*24))//nsteps_h
            power=Ptot[i]
            price = price_tou[hindex]
            # the power should divide by 1000
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
                overshoot.append(np.array([float((TZone[i] -273.15) - T_upper[hindex]), 0.0]).max())
                undershoot.append(np.array([float(T_lower[hindex] - (TZone[i]-273.15)), 0.0]).max())

            penalty.append(sum(np.array(overshoot)) + sum(np.array(undershoot)))
        
            # action changes
            delta_action.append(abs(u[i] - u_prev[i]))

            # sum up for rewards
            rewards.append(rw_func(energy_cost[-1], penalty[-1], delta_action[-1]))

        return np.array([energy_cost, penalty, delta_action, rewards]).transpose()

    #### get rewards
    #================================================================================
    rewards_base = get_rewards(measurement_base['fcu.uFan'].values, measurement_base['PTot'].values,measurement_base['TRoo'].values,price_tou)
    rewards_mpc = get_rewards(measurement_mpc['fcu.uFan'].values, measurement_mpc['PTot'].values,measurement_mpc['TRoo'].values,price_tou)

    rewards_base = pd.DataFrame(rewards_base,columns=[['ene_cost','penalty', 'delta_action', 'rewards']])
    rewards_mpc = pd.DataFrame(rewards_mpc,columns=[['ene_cost','penalty','delta_action', 'rewards']])

    # get rewards - DRL - we can either read from training results or 
    # recalculate using the method for mpc and baseline (very time consuming for multi-epoch training)
    ### ===============
    mpc_rbc_rewards={'base': {'rewards':list(rewards_base['rewards'].sum()),
                            'ene_cost': list(rewards_base['ene_cost'].sum()),
                            'total_temp_violation': list(rewards_base['penalty'].sum()/4),
                            'max_temp_violation': list(rewards_base['penalty'].max()),
                            'temp_violation_squared': list((rewards_base['penalty']**2).sum()),
                            'delta_action_sqaured': list((rewards_base['delta_action']**2).sum())},
                    'mpc': {'rewards': list(rewards_mpc['rewards'].sum()),
                            'ene_cost': list(rewards_mpc['ene_cost'].sum()),
                            'total_temp_violation': list(rewards_mpc['penalty'].sum()/4),
                            'max_temp_violation': list(rewards_mpc['penalty'].max()),
                            'temp_violation_squared': list((rewards_mpc['penalty']**2).sum()),
                            'delta_action_sqaured': list((rewards_mpc['delta_action']**2).sum())},
                    }
    with open(os.path.join(path, 'mpc_rewards.json'), 'w') as outfile:
        json.dump(mpc_rbc_rewards, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str)
    args = parser.parse_args()

    # simulation setup
    ts = 201*24*3600.  # +13*24*3600
    nday = 7
    period = nday*24*3600.
    te = ts + period
    dt = 15*60.

    measurement_names = ['time', 'TRoo', 'TOut', 'PTot', 'fcu.uFan', 'm_flow_in']
    
    # simulate baseline
    measurement_base = simulate_baseline(ts, te, measurement_names)

    ################################################
    ##           MPC Final Simulation
    ## =============================================
    root_dir = args.root_dir
    u_opt_files = find_all_files(root_dir)
    weights = [100., 1., 10.]

    for file in u_opt_files:
        #extract path
        path = os.path.dirname(file)
        #extract PH
        PH = int(re.findall("PH=\d+", file)[0].split("=")[1])
        measurement_mpc = simulate_mpc(file, ts, te, dt, measurement_names)
        plot(ts, te, dt, nday, PH, measurement_base, measurement_mpc, path)
        calculate_metrics(ts, te, dt, measurement_base, measurement_mpc, weights, path)
