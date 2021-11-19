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
ts = 204*24*3600.#+13*24*3600
nday = 7
period = nday*24*3600.
te = ts + period
dt = 15*60.
nsteps_h = int(3600//dt)

# setup for DRL test
nActions = 51
alpha = 1
nepochs = 50

# define some filters to save simulation time using fmu
measurement_names = ['time','PHVAC','PBoiGas','TRooAirSou','TRooAirEas','TRooAirNor','TRooAirWes','TRooAirCor','TRooAirDevTot','EHVACTot','conAHU.TSupSet','conAHU.TSup','uTSupSet']

##########################################
##          Baseline Simulation
## =========================================
# DEFINE MODEL
# ------------
baseline = load_fmu('FiveZoneVAVBaseline.fmu')

## fmu settings
options = baseline.simulate_options()
options['ncp'] = 5000
options['initialize'] = True
options['result_handling'] = 'memory'
options['filter'] = measurement_names[:-1]

## construct optimal input for fmu
res_base = baseline.simulate(start_time = ts,
                    final_time = te, 
                    options = options)

print ("Finish baseline simulation")

###############################################################
##              DRL final run: Discrete: v0-dqn
##===========================================================
# get actions from the last epoch
v0_dqn_case = './dqn_results_d204_a_100'
actions= np.load(v0_dqn_case+'/his_act.npy')
u_opt = 12+6*np.array(actions[:-1])/float(nActions-1)+273.15
print (u_opt)

### 1- Load virtual building model
hvac = load_fmu('FiveZoneVAV.fmu')

## fmu settings
options = hvac.simulate_options()
options['ncp'] = 100
options['initialize'] = True
options['result_handling'] = 'memory'
options['filter'] = measurement_names
res_dqn_v0 = []

## construct optimal input for fmu
# main loop - do step
t = ts
i = 0
while t < te:
    u = u_opt[i]
    u_traj = np.transpose(np.vstack(([t,t+dt],[u,u])))
    input_object = ("uTSupSet",u_traj)
    ires = hvac.simulate(start_time = t,
                final_time = t+dt, 
                options = options,
                input = input_object)
    res_dqn_v0.append(ires)
    t += dt 
    i += 1
    options['initialize'] = False



##############################################################
#           Compare DRL with Baseline
# =============================================================

# read measurements
measurement_base = {}
measurement_dqn_v0 = {}
for name in measurement_names[:-1]:
    measurement_base[name] = res_base[name]
# get dqn_v0 results
for name in measurement_names:
    value_name_dqn_v0=[]
    for ires in res_dqn_v0:
      value_name_dqn_v0 += list(ires[name])
    measurement_dqn_v0[name] = np.array(value_name_dqn_v0)

## simulate baseline
occ_start = 7
occ_end = 19
tim = np.arange(ts,te,dt)
T_upper = np.array([30.0 for i in tim])
#T_upper[occ_start*4:(occ_end-1)*4] = 24.5
T_lower = np.array([12.0 for i in tim])
#T_lower[occ_start*4:(occ_end-1)*4] = 23.5
for i in range(nday):
  T_upper[24*nsteps_h*i+occ_start*nsteps_h:24*nsteps_h*i+(occ_end-1)*nsteps_h] = 24.5
  T_lower[24*nsteps_h*i+occ_start*nsteps_h:24*nsteps_h*i+(occ_end-1)*nsteps_h] = 23.5

# price_tou = [0.0640, 0.0640, 0.0640, 0.0640, 
        # 0.0640, 0.0640, 0.0640, 0.0640, 
        # 0.1391, 0.1391, 0.1391, 0.1391, 
        # 0.3548, 0.3548, 0.3548, 0.3548, 
        # 0.3548, 0.3548, 0.1391, 0.1391, 
        # 0.1391, 0.1391, 0.1391, 0.0640]*nday

price_tou = [1]*24*nday

xticks=np.arange(ts,te,12*3600)
xticks_label = np.arange(0,24*nday,12)

plt.figure(figsize=(16,28))
# plt.subplot(411)
# plt.step(np.arange(ts, te, 3600.),price_tou, where='post')
# plt.xticks(xticks,[])
# plt.grid(True)
# plt.ylabel('Price ($/kW)')

plt.subplot(711)
plt.plot(measurement_base['time'], measurement_base['conAHU.TSup']-273.15,'b--',label='Baseline')
plt.plot(measurement_base['time'], measurement_base['conAHU.TSupSet']-273.15,'b-')
plt.plot(measurement_dqn_v0['time'], measurement_dqn_v0['conAHU.TSup']-273.15,'r--',label='DQN')
plt.plot(measurement_dqn_v0['time'], measurement_dqn_v0['conAHU.TSupSet']-273.15,'r-')
plt.grid(True)
plt.xticks(xticks,[])
plt.legend()
plt.ylabel('Supply air temperature [C]')

plt.subplot(712)
plt.plot(measurement_base['time'], measurement_base['TRooAirSou']-273.15,'b-',label='Baseline')
plt.plot(measurement_dqn_v0['time'],  measurement_dqn_v0['TRooAirSou']-273.15,'r--',label='DQN')
plt.plot(tim,T_upper, 'g-.', lw=1,label='Bounds')
plt.plot(tim,T_lower, 'g-.', lw=1)
plt.grid(True)
plt.xticks(xticks,[])
plt.legend()
plt.ylabel('South Zone Temperature [C]')
plt.ylim(22, 26)

plt.subplot(713)
plt.plot(measurement_base['time'], measurement_base['TRooAirEas']-273.15,'b-',label='Baseline')
plt.plot(measurement_dqn_v0['time'],  measurement_dqn_v0['TRooAirEas']-273.15,'r--',label='DQN')
plt.plot(tim,T_upper, 'g-.', lw=1,label='Bounds')
plt.plot(tim,T_lower, 'g-.', lw=1)
plt.grid(True)
plt.xticks(xticks,[])
plt.legend()
plt.ylabel('East Zone Temperature [C]')
plt.ylim(22, 26)  

plt.subplot(714)
plt.plot(measurement_base['time'], measurement_base['TRooAirWes']-273.15,'b-',label='Baseline')
plt.plot(measurement_dqn_v0['time'],  measurement_dqn_v0['TRooAirWes']-273.15,'r--',label='DQN')
plt.plot(tim,T_upper, 'g-.', lw=1,label='Bounds')
plt.plot(tim,T_lower, 'g-.', lw=1)
plt.grid(True)
plt.xticks(xticks,[])
plt.legend()
plt.ylabel('West Zone Temperature [C]')
plt.ylim(22, 26)  

plt.subplot(715)
plt.plot(measurement_base['time'], measurement_base['TRooAirNor']-273.15,'b-',label='Baseline')
plt.plot(measurement_dqn_v0['time'],  measurement_dqn_v0['TRooAirNor']-273.15,'r--',label='DQN')
plt.plot(tim,T_upper, 'g-.', lw=1,label='Bounds')
plt.plot(tim,T_lower, 'g-.', lw=1)
plt.grid(True)
plt.xticks(xticks,[])
plt.legend()
plt.ylabel('North Zone Temperature [C]')
plt.ylim(22, 26)  

plt.subplot(716)
plt.plot(measurement_base['time'], measurement_base['TRooAirCor']-273.15,'b-',label='Baseline')
plt.plot(measurement_dqn_v0['time'],  measurement_dqn_v0['TRooAirCor']-273.15,'r--',label='DQN')
plt.plot(tim,T_upper, 'g-.', lw=1,label='Bounds')
plt.plot(tim,T_lower, 'g-.', lw=1)
plt.grid(True)
plt.xticks(xticks,[])
plt.legend()
plt.ylabel('Core Zone Temperature [C]')
plt.ylim(22, 26)  

plt.subplot(717)
plt.plot(measurement_base['time'], measurement_base['PHVAC'],'b-',label='Baseline')
plt.plot(measurement_dqn_v0['time'], measurement_dqn_v0['PHVAC'],'r--',label='DQN')
plt.grid(True)
plt.xticks(xticks,xticks_label)
plt.ylabel('Total [W]')
plt.legend()
plt.savefig('drl_comparison_interpolate.pdf')


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
measurement_dqn_v0 = pd.DataFrame(measurement_dqn_v0,index=measurement_dqn_v0['time'])

tim_intp = np.arange(ts,te,dt)
measurement_base = interpolate_dataframe(measurement_base[['PHVAC','TRooAirSou','TRooAirEas','TRooAirNor','TRooAirWes','TRooAirCor']],tim_intp)
measurement_dqn_v0 = interpolate_dataframe(measurement_dqn_v0[['PHVAC','TRooAirSou','TRooAirEas','TRooAirNor','TRooAirWes','TRooAirCor']],tim_intp)

#measurement_base.to_csv('measurement_base.csv')
#measurement_dqn_v0.to_csv('measurement_dqn_v0.csv')

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
    res = penalty * 50000.0 + cost*500
    return res
    
def get_rewards(PHVAC,TZone,price_tou,alpha):
    n= len(PHVAC)
    energy_cost = []
    penalty = []
    rewards = []
    alpha_up = alpha
    alpha_low = alpha
    for i in range(n):
        # assume 1 step is 15 minutes and data starts from hour 0
        hindex = (i%(nsteps_h*24))//nsteps_h
        power=PHVAC[i]
        price = price_tou[hindex]
        # the power should divide by 1000
        energy_cost.append(power/1000./nsteps_h*price)
        # zone temperature penalty
        number_zone = 1
        # zone temperature bounds - need check with the high-fidelty model
        T_upper = np.array([30.0 for j in range(24)])
        T_upper[7:19] = 24.5
        T_lower = np.array([12.0 for j in range(24)])
        T_lower[7:19] = 23.5
        overshoot = []
        undershoot = []
        for k in range(number_zone):
            overshoot.append(np.array([float((TZone[k][i] -273.15) - T_upper[hindex]), 0.0]).max())
            undershoot.append(np.array([float(T_lower[hindex] - (TZone[k][i]-273.15)), 0.0]).max())
        penalty.append(alpha_up*sum(np.array(overshoot)) + alpha_low*sum(np.array(undershoot)))
        # sum up for rewards
        rewards.append(rw_func(energy_cost[-1], penalty[-1]))
    return -np.array([energy_cost, penalty, rewards]).transpose()

#### get rewards
#================================================================================
rewards_base = get_rewards(measurement_base['PHVAC'].values,measurement_base[['TRooAirSou','TRooAirEas','TRooAirNor','TRooAirWes','TRooAirCor']].values.T,price_tou,alpha)

rewards_base = pd.DataFrame(rewards_base,columns=[['ene_cost','penalty','rewards']])

# get rewards - DRL - we can either read from training results or 
# recalculate using the method for mpc and baseline (very time consuming for multi-epoch training)
rewards_dqn_v0_hist = np.load(v0_dqn_case+'/his_rew.npy')
rewards_dqn_v0 = []

rewards_dqn_v0_hist=rewards_dqn_v0_hist[128:128+672*nepochs]
rewards_dqn_v0_hist=rewards_dqn_v0_hist.reshape(50,672)

for epoch in range(nepochs):
    rewards_dqn_v0 += list(rewards_dqn_v0_hist[epoch,:])
    # rewards_dqn_v0 += list(rewards_dqn_v0_hist[epoch,:-1])
rewards_dqn_v0 = pd.DataFrame(np.array(rewards_dqn_v0),columns=['rewards'])
print (rewards_dqn_v0)



# plot rewards with moving windows - epoch-by-epoch
moving = nday*24*3600.//dt 
rewards_moving_base = rewards_base['rewards'].groupby(rewards_base.index//moving).sum()
rewards_moving_dqn_v0 = rewards_dqn_v0['rewards'].groupby(rewards_dqn_v0.index//moving).sum()


plt.figure(figsize=(9,6))
plt.plot(list(rewards_moving_base.values)*nepochs,'b-',label='RBC')
plt.plot(rewards_moving_dqn_v0.values,'r--',lw=1,label='DQN')

plt.ylabel('rewards')
plt.xlabel('epoch')
plt.grid(True)
plt.legend()
plt.savefig('rewards_epoch.pdf')
#plt.savefig('rewards_epoch.png')


# The following codes are only for comparing occupied control performance
#===========================================================================
#flag = np.logical_or(rewards_dqn_v0_last.index%96//4<7,rewards_dqn_v0_last.index%96//4>=19)
#rewards_dqn_v0_last.loc[flag,'ene_cost'] = 0
#print rewards_dqn_v0_last['ene_cost'].sum()

# save total energy cost, violations etc using the final episode for DRL
rewards_dqn_v0_last = get_rewards(measurement_dqn_v0['PHVAC'].values,measurement_dqn_v0[['TRooAirSou','TRooAirEas','TRooAirNor','TRooAirWes','TRooAirCor']].values.T,price_tou,alpha)
rewards_dqn_v0_last = pd.DataFrame(rewards_dqn_v0_last,columns=[['ene_cost','penalty','rewards']])



comparison={'base':{'energy_cost':list(rewards_base['ene_cost'].sum()),
                    'temp_violation':list(rewards_base['penalty'].sum())},
            'dqn':{'energy_cost':list(rewards_dqn_v0_last['ene_cost'].sum()),
                    'temp_violation':list(rewards_dqn_v0_last['penalty'].sum())}
                    }
with open('comparison_epoch.json', 'w') as outfile:
    json.dump(comparison, outfile)

