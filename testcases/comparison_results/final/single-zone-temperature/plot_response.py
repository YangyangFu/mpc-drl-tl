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
ts = 212*24*3600.#+13*24*3600
nday = 7
period = nday*24*3600.
te = ts + period
dt = 15*60.
nsteps_h = int(3600//dt)

# setup for DRL test
nActions = 37
alpha = 1
nepochs = 500

# define some filters to save simulation time using fmu
measurement_names = ['time','TRoo','TOut','PTot','hvac.uFan','hvac.fanSup.m_flow_in', 'senTSetRooCoo.y', 'CO2Roo']

##########################################
##          Baseline Simulation
## =========================================
# DEFINE MODEL
# ------------
baseline = load_fmu('SingleZoneTemperatureBaseline.fmu')

## fmu settings
options = baseline.simulate_options()
options['ncp'] = 5000.
options['initialize'] = True
options['result_handling'] = 'memory'
options['filter'] = measurement_names

## construct optimal input for fmu
res_base = baseline.simulate(start_time = ts,
                    final_time = te, 
                    options = options)

################################################
##           MPC Final Simulation
## =============================================

# read optimal control inputs
with open('./mpc/u_opt.json') as f:
  opt = json.load(f)

t_opt = opt['t_opt']
u_opt = opt['u_opt']

print(len(t_opt))
print(len(u_opt))

### 1- Load virtual building model
hvac = load_fmu('SingleZoneTemperature.fmu')

## fmu settings
options = hvac.simulate_options()
options['ncp'] = 100.
options['initialize'] = True
options['result_handling'] = 'memory'
options['filter'] = measurement_names
res_mpc = []

# main loop - do step
t = ts
i = 0
while t < te:
    u = u_opt[i]
    u_traj = np.transpose(np.vstack(([t,t+dt],[u,u])))
    input_object = ("TSetCoo",u_traj)
    ires = hvac.simulate(start_time = t,
                final_time = t+dt, 
                options = options,
                input = input_object)
    res_mpc.append(ires)

    t += dt 
    i += 1
    options['initialize'] = False


###############################################################
##              DRL final run: Discrete: v1-dqn
##===========================================================
# get actions from the last epoch
v1_dqn_case = './v1-dqn'
actions= np.load(v1_dqn_case+'/his_act.npy')
u_opt = np.array(actions[-1,:-1])/float(nActions-1)*(30.-12.)+12+273.15
print (u_opt)

## fmu settings
hvac.reset()
options = hvac.simulate_options()
options['ncp'] = 100.
options['initialize'] = True
options['result_handling'] = 'memory'
options['filter'] = measurement_names
res_dqn_v1 = []

## construct optimal input for fmu
# main loop - do step
t = ts
i = 0
while t < te:
    u = u_opt[i]
    u_traj = np.transpose(np.vstack(([t,t+dt],[u,u])))
    input_object = ("TSetCoo",u_traj)
    ires = hvac.simulate(start_time = t,
                final_time = t+dt, 
                options = options,
                input = input_object)
    res_dqn_v1.append(ires)

    t += dt 
    i += 1
    options['initialize'] = False

###############################################################
##              DRL final run: Discrete: v1-sac
##===========================================================
# get actions from the last epoch
v1_sac_case = './v1-sac'
actions= np.load(v1_sac_case+'/his_act.npy')
u_opt = np.array(actions[-1,:-1])/float(nActions-1)*(30.-12.)+12+273.15
print (u_opt)

## fmu settings
hvac.reset()
options = hvac.simulate_options()
options['ncp'] = 100.
options['initialize'] = True
options['result_handling'] = 'memory'
options['filter'] = measurement_names
res_sac_v1 = []

## construct optimal input for fmu
# main loop - do step
t = ts
i = 0
while t < te:
    u = u_opt[i]
    u_traj = np.transpose(np.vstack(([t,t+dt],[u,u])))
    input_object = ("TSetCoo",u_traj)
    ires = hvac.simulate(start_time = t,
                final_time = t+dt, 
                options = options,
                input = input_object)
    res_sac_v1.append(ires)

    t += dt 
    i += 1
    options['initialize'] = False

################################################################
##           Compare MPC/DRL with Baseline
## =============================================================

# read measurements
measurement_mpc = {}
measurement_base = {}
measurement_dqn_v1 = {}
measurement_sac_v1 = {}

for name in measurement_names:
    measurement_base[name] = res_base[name]
    # get mpc results
    value_name_mpc=[]
    for ires in res_mpc:
      value_name_mpc += list(ires[name])
    measurement_mpc[name] = np.array(value_name_mpc)
    # get dqn_v1 results
    value_name_dqn_v1=[]
    for ires in res_dqn_v1:
      value_name_dqn_v1 += list(ires[name])
    measurement_dqn_v1[name] = np.array(value_name_dqn_v1)
    # get sac_v1 results
    value_name_sac_v1=[]
    for ires in res_sac_v1:
      value_name_sac_v1 += list(ires[name])
    measurement_sac_v1[name] = np.array(value_name_sac_v1)

## simulate baseline
occ_start = 7
occ_end = 19
tim = np.arange(ts,te,dt)
T_upper = np.array([30.0 for i in tim])
#T_upper[occ_start*4:(occ_end-1)*4] = 26.0
T_lower = np.array([12.0 for i in tim])
#T_lower[occ_start*4:(occ_end-1)*4] = 22.0
for i in range(nday):
  T_upper[24*nsteps_h*i+occ_start*nsteps_h:24*nsteps_h*i+(occ_end-1)*nsteps_h] = 26.0
  T_lower[24*nsteps_h*i+occ_start*nsteps_h:24*nsteps_h*i+(occ_end-1)*nsteps_h] = 22.

price_tou = [0.0640, 0.0640, 0.0640, 0.0640, 
        0.0640, 0.0640, 0.0640, 0.0640, 
        0.1391, 0.1391, 0.1391, 0.1391, 
        0.3548, 0.3548, 0.3548, 0.3548, 
        0.3548, 0.3548, 0.1391, 0.1391, 
        0.1391, 0.1391, 0.1391, 0.0640]*nday

xticks=np.arange(ts,te,12*3600)
xticks_label = np.arange(0,24*nday,12)

plt.figure(figsize=(16,12))
plt.subplot(411)
plt.step(np.arange(ts, te, 3600.),price_tou, where='post')
plt.xticks(xticks,[])
plt.grid(True)
plt.ylabel('Price ($/kW)')

plt.subplot(412)
plt.subplot(412)
plt.plot(measurement_base['time'], measurement_base['senTSetRooCoo.y']-273.15,'b-',label='Baseline')
plt.plot(measurement_mpc['time'], measurement_mpc['senTSetRooCoo.y']-273.15,'b--',label='MPC')
plt.plot(measurement_dqn_v1['time'], measurement_dqn_v1['senTSetRooCoo.y']-273.15,'r--',label='DQN')
plt.plot(measurement_sac_v1['time'], measurement_sac_v1['senTSetRooCoo.y']-273.15,'y--',label='SAC')
plt.grid(True)
plt.xticks(xticks,[])
plt.legend()
plt.ylabel('Cooling Setpoint [C]')

plt.subplot(413)
plt.plot(measurement_base['time'], measurement_base['TRoo']-273.15,'b-',label='Baseline')
plt.plot(measurement_mpc['time'],  measurement_mpc['TRoo']-273.15,'b--',label='MPC')
plt.plot(measurement_dqn_v1['time'],  measurement_dqn_v1['TRoo']-273.15,'r--',label='DQN')
plt.plot(measurement_sac_v1['time'],  measurement_sac_v1['TRoo']-273.15,'y--',label='SAC')
plt.plot(tim,T_upper, 'g-.', lw=1,label='Bounds')
plt.plot(tim,T_lower, 'g-.', lw=1)
plt.grid(True)
plt.xticks(xticks,[])
plt.legend()
plt.ylabel('Room Temperature [C]')

plt.subplot(414)
plt.plot(measurement_base['time'], measurement_base['PTot'],'b-',label='Baseline')
plt.plot(measurement_mpc['time'], measurement_mpc['PTot'],'b--',label='MPC')
plt.plot(measurement_dqn_v1['time'], measurement_dqn_v1['PTot'],'r--',label='DQN')
plt.plot(measurement_sac_v1['time'], measurement_sac_v1['PTot'],'y--',label='SAC')
plt.grid(True)
plt.xticks(xticks,xticks_label)
plt.ylabel('Total [W]')
plt.savefig('mpc-drl.pdf')
plt.savefig('mpc-drl.png')

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
measurement_dqn_v1 = pd.DataFrame(measurement_dqn_v1,index=measurement_dqn_v1['time'])
measurement_sac_v1 = pd.DataFrame(measurement_sac_v1,index=measurement_sac_v1['time'])

tim_intp = np.arange(ts,te,dt)
measurement_base = interpolate_dataframe(measurement_base[['PTot','TRoo']],tim_intp)
measurement_mpc = interpolate_dataframe(measurement_mpc[['PTot','TRoo']],tim_intp)
measurement_dqn_v1 = interpolate_dataframe(measurement_dqn_v1[['PTot','TRoo']],tim_intp)
measurement_sac_v1 = interpolate_dataframe(measurement_sac_v1[['PTot','TRoo']],tim_intp)

#measurement_base.to_csv('measurement_base.csv')
#measurement_mpc.to_csv('measurement_mpc.csv')
#measurement_dqn_v1.to_csv('measurement_dqn_v1.csv')
#measurement_sac_v1.to_csv('measurement_sac_v1.csv')

def rw_func(cost, penalty):
    if ( not hasattr(rw_func,'x')  ):
        rw_func.x = 0
        rw_func.y = 0

    #cost = cost[0]
    #penalty = penalty[0]

    if rw_func.x > cost:
        rw_func.x = cost
    if rw_func.y > penalty:
        rw_func.y = penalty

    #print("rw_func-cost-min=", rw_func.x, ". penalty-min=", rw_func.y)
    #res = penalty * 10.0
    #res = penalty * 300.0 + cost*1e4
    res = penalty * 500.0 + cost*5e3
    
    return res
def get_rewards(Ptot,TZone,price_tou,alpha):
    n= len(Ptot)
    energy_cost = []
    penalty = []
    rewards = []

    alpha_up = alpha
    alpha_low = alpha

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
        T_upper[7:19] = 26.0
        T_lower = np.array([12.0 for j in range(24)])
        T_lower[7:19] = 22.0

        overshoot = []
        undershoot = []
        for k in range(number_zone):
            overshoot.append(np.array([float((TZone[i] -273.15) - T_upper[hindex]), 0.0]).max())
            undershoot.append(np.array([float(T_lower[hindex] - (TZone[i]-273.15)), 0.0]).max())

        penalty.append(alpha_up*sum(np.array(overshoot)) + alpha_low*sum(np.array(undershoot)))
    
        # sum up for rewards
        rewards.append(rw_func(energy_cost[-1], penalty[-1]))

    return -np.array([energy_cost, penalty, rewards]).transpose()

#### get rewards
#================================================================================
rewards_base = get_rewards(measurement_base['PTot'].values,measurement_base['TRoo'].values,price_tou,alpha)
rewards_mpc = get_rewards(measurement_mpc['PTot'].values,measurement_mpc['TRoo'].values,price_tou,alpha)

rewards_base = pd.DataFrame(rewards_base,columns=[['ene_cost','penalty','rewards']])
rewards_mpc = pd.DataFrame(rewards_mpc,columns=[['ene_cost','penalty','rewards']])

# get rewards - DRL - we can either read from training results or 
# recalculate using the method for mpc and baseline (very time consuming for multi-epoch training)
rewards_dqn_v1_hist = np.load(v1_dqn_case+'/his_rew.npy')
rewards_dqn_v1 = []
# for zone 1
for epoch in range(nepochs):
    rewards_dqn_v1 += list(rewards_dqn_v1_hist[epoch,:-1])

rewards_dqn_v1 = pd.DataFrame(np.array(rewards_dqn_v1),columns=['rewards'])
print (rewards_dqn_v1)


rewards_sac_v1_hist = np.load(v1_sac_case+'/his_rew.npy')
rewards_sac_v1 = []
# for zone 1
for epoch in range(nepochs):
    rewards_sac_v1 += list(rewards_sac_v1_hist[epoch,:-1])

rewards_sac_v1 = pd.DataFrame(np.array(rewards_sac_v1),columns=['rewards'])
print (rewards_sac_v1)

# plot rewards with moving windows - epoch-by-epoch
moving = nday*24*3600.//dt 
rewards_moving_base = rewards_base['rewards'].groupby(rewards_base.index//moving).sum()
rewards_moving_mpc = rewards_mpc['rewards'].groupby(rewards_mpc.index//moving).sum()
rewards_moving_dqn_v1 = rewards_dqn_v1['rewards'].groupby(rewards_dqn_v1.index//moving).sum()
rewards_moving_sac_v1 = rewards_sac_v1['rewards'].groupby(rewards_sac_v1.index//moving).sum()

plt.figure(figsize=(9,6))
plt.plot(list(rewards_moving_base.values)*nepochs,'b-',label='RBC')
plt.plot(list(rewards_moving_mpc.values)*nepochs,'b--',label='MPC')
plt.plot(rewards_moving_dqn_v1.values,'r--',lw=1,label='DQN')
plt.plot(rewards_moving_sac_v1.values,'y--',lw=1,label='SAC')
plt.ylabel('rewards')
plt.xlabel('epoch')
plt.grid(True)
plt.legend()
plt.savefig('rewards_epoch.pdf')
plt.savefig('rewards_epoch.png')


# The following codes are only for comparing occupied control performance
#===========================================================================
#flag = np.logical_or(rewards_dqn_v1_last.index%96//4<7,rewards_dqn_v1_last.index%96//4>=19)
#rewards_dqn_v1_last.loc[flag,'ene_cost'] = 0
#print rewards_dqn_v1_last['ene_cost'].sum()

# save total energy cost, violations etc using the final episode for DRL
rewards_dqn_v1_last = get_rewards(measurement_dqn_v1['PTot'].values,measurement_dqn_v1['TRoo'].values,price_tou,alpha)
rewards_dqn_v1_last = pd.DataFrame(rewards_dqn_v1_last,columns=[['ene_cost','penalty','rewards']])

rewards_sac_v1_last = get_rewards(measurement_sac_v1['PTot'].values,measurement_sac_v1['TRoo'].values,price_tou,alpha)
rewards_sac_v1_last = pd.DataFrame(rewards_sac_v1_last,columns=[['ene_cost','penalty','rewards']])

comparison={'base':{'energy_cost':list(rewards_base['ene_cost'].sum()),
                    'temp_violation':list(rewards_base['penalty'].sum())},
            'mpc':{'energy_cost':list(rewards_mpc['ene_cost'].sum()),
                    'temp_violation':list(rewards_mpc['penalty'].sum())},
            'dqn':{'energy_cost':list(rewards_dqn_v1_last['ene_cost'].sum()),
                    'temp_violation':list(rewards_dqn_v1_last['penalty'].sum())},
            'sac':{'energy_cost':list(rewards_sac_v1_last['ene_cost'].sum()),
                    'temp_violation':list(rewards_sac_v1_last['penalty'].sum())}
                    }
with open('comparison_epoch.json', 'w') as outfile:
    json.dump(comparison, outfile)

