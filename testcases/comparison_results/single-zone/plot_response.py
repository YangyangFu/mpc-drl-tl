from __future__ import print_function
from __future__ import absolute_import, division

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import json
# load testbed
from pyfmi import load_fmu

COLORS = (
    [
        # personal color
        '#313695',  # DARK BLUE
        '#74add1',  # LIGHT BLUE
        '#f46d43',  # ORANGE
        '#4daf4a',  # GREEN
        '#984ea3',  # PURPLE
        '#f781bf',  # PINK
        '#ffc832',  # YELLOW
        '#000000',  # BLACK
    ])
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
options['ncp'] = 5000
options['initialize'] = True
options['result_handling'] = 'memory'
options['filter'] = measurement_names
baseline.set("zon.roo.T_start", 273.15+25)
## construct optimal input for fmu
res_base = baseline.simulate(start_time = ts,
                    final_time = te, 
                    options = options)

################################################
##           MPC Final Simulation
## =============================================

# read optimal control inputs
with open('./mpc/R2/PH=96/u_opt.json') as f:
  opt = json.load(f)

t_opt = opt['t_opt']
u_opt = opt['u_opt']

print(len(t_opt))
print(len(u_opt))

### 1- Load virtual building model
hvac = load_fmu('SingleZoneFCU.fmu')

## fmu settings
options = hvac.simulate_options()
options['ncp'] = 100
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
    hvac.set('uFan',u[0])
    ires = hvac.simulate(start_time = t,
                final_time = t+dt, 
                options = options)
    res_mpc.append(ires)

    t += dt 
    i += 1
    options['initialize'] = False


###############################################################
##              DRL final run: ddqn-seed2
##===========================================================
# get actions from the last epoch
ddqn_case = './DRL-R2/ddqn_seed2/'
with open(ddqn_case+'u_opt.json') as f:
  u_opt = json.load(f)

## fmu settings
hvac.reset()
options = hvac.simulate_options()
options['ncp'] = 100
options['initialize'] = True
options['result_handling'] = 'memory'
options['filter'] = measurement_names
res_ddqn = []
hvac.set("zon.roo.T_start", 273.15+25)
## construct optimal input for fmu
# main loop - do step
t = ts
i = 0
while t < te:
    u = u_opt[i]
    hvac.set('uFan',u[0])
    ires = hvac.simulate(start_time = t,
                final_time = t+dt, 
                options = options)
    res_ddqn.append(ires)

    t += dt 
    i += 1
    options['initialize'] = False

###############################################################
##              DRL final run: sac_seed0
##===========================================================
# get actions from the last epoch
sac_case = './DRL-R2/sac_seed0/'
with open(sac_case+'u_opt.json') as f:
  u_opt = json.load(f)

## fmu settings
hvac.reset()
options = hvac.simulate_options()
options['ncp'] = 100
options['initialize'] = True
options['result_handling'] = 'memory'
options['filter'] = measurement_names
res_sac = []
hvac.set("zon.roo.T_start", 273.15+25)
## construct optimal input for fmu
# main loop - do step
t = ts
i = 0
while t < te:
    u = u_opt[i]
    hvac.set('uFan', u[0])
    ires = hvac.simulate(start_time = t,
                final_time = t+dt, 
                options = options)
    res_sac.append(ires)

    t += dt 
    i += 1
    options['initialize'] = False

###############################################################
##              DRL final run: ppo_seed3
##===========================================================
# get actions from the last epoch
ppo_case = './DRL-R2/ppo_seed3/'
with open(ppo_case+'u_opt.json') as f:
  u_opt = json.load(f)

## fmu settings
hvac.reset()
options = hvac.simulate_options()
options['ncp'] = 100
options['initialize'] = True
options['result_handling'] = 'memory'
options['filter'] = measurement_names
res_ppo = []
hvac.set("zon.roo.T_start", 273.15+25)
## construct optimal input for fmu
# main loop - do step
t = ts
i = 0
while t < te:
    u = u_opt[i]
    hvac.set('uFan', u[0])
    ires = hvac.simulate(start_time=t,
                         final_time=t+dt,
                         options=options)
    res_ppo.append(ires)

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
measurement_sac = {}
measurement_ppo = {}

for name in measurement_names:
    measurement_base[name] = res_base[name]
    # get mpc results
    value_name_mpc=[]
    for ires in res_mpc:
      value_name_mpc += list(ires[name])
    measurement_mpc[name] = np.array(value_name_mpc)
    # get ddqn results
    value_name_ddqn=[]
    for ires in res_ddqn:
      value_name_ddqn += list(ires[name])
    measurement_ddqn[name] = np.array(value_name_ddqn)
    # get sac results
    value_name_sac=[]
    for ires in res_sac:
      value_name_sac += list(ires[name])
    measurement_sac[name] = np.array(value_name_sac)
    # get ppo
    value_name_ppo=[]
    for ires in res_ppo:
      value_name_ppo += list(ires[name])
    measurement_ppo[name] = np.array(value_name_ppo)

## simulate baseline
occ_start = 8
occ_end = 18
tim = np.arange(ts,te,dt)
T_upper = np.array([30.0 for i in tim])
T_lower = np.array([12.0 for i in tim])
for i in range(nday):
  T_upper[24*nsteps_h*i+occ_start*nsteps_h:24*nsteps_h*i+occ_end*nsteps_h] = 26.0
  T_lower[24*nsteps_h*i+occ_start*nsteps_h:24*nsteps_h*i+occ_end*nsteps_h] = 22.0

price_tou = [0.02987, 0.02987, 0.02987, 0.02987,
            0.02987, 0.02987, 0.04667, 0.04667,
            0.04667, 0.04667, 0.04667, 0.04667,
            0.15877, 0.15877, 0.15877, 0.15877,
            0.15877, 0.15877, 0.15877, 0.04667,
            0.04667, 0.04667, 0.02987, 0.02987]*nday

xticks=np.arange(ts,te+1,12*3600)
xticks_label = np.arange(0,24*nday+1,12)

plt.figure(figsize=(16,12))
plt.subplot(411)
plt.step(np.arange(ts, te, 3600.),price_tou, where='post',c='k')
plt.xticks(xticks,[])
plt.grid(True)
plt.ylabel('Price ($/kW)')

plt.subplot(412)
plt.plot(measurement_base['time'], measurement_base['fcu.uFan'], c=COLORS[0], label='RBC')
plt.plot(measurement_mpc['time'], measurement_mpc['fcu.uFan'],c=COLORS[1], label='MPC')
plt.plot(measurement_ddqn['time'], measurement_ddqn['fcu.uFan'],c=COLORS[2], label='DDQN')
plt.plot(measurement_ppo['time'], measurement_ppo['fcu.uFan'],c=COLORS[3], label='PPO')
plt.plot(measurement_sac['time'], measurement_sac['fcu.uFan'],c=COLORS[5],label='SAC')
plt.grid(True)
plt.xticks(xticks,[])
plt.legend(fancybox=True, framealpha=0.5)
plt.ylabel('Fan Speed')

plt.subplot(413)
plt.plot(measurement_base['time'], measurement_base['TRoo']-273.15,c=COLORS[0], label='RBC')
plt.plot(measurement_mpc['time'],  measurement_mpc['TRoo']-273.15,c=COLORS[1], label='MPC')
plt.plot(measurement_ddqn['time'],  measurement_ddqn['TRoo']-273.15,c=COLORS[2], label='DDQN')
plt.plot(measurement_ppo['time'],  measurement_ppo['TRoo']-273.15,c=COLORS[3], label='PPO')
plt.plot(measurement_sac['time'],  measurement_sac['TRoo']-273.15,c=COLORS[5],label='SAC')
plt.plot(tim,T_upper, 'k-.', lw=1,label='Bounds')
plt.plot(tim,T_lower, 'k-.', lw=1)
plt.grid(True)
plt.xticks(xticks,[])
plt.legend(fancybox=True, framealpha=0.5)
plt.ylabel('Room Temperature [C]')

plt.subplot(414)
plt.plot(measurement_base['time'], measurement_base['PTot'], c=COLORS[0], label='RBC')
plt.plot(measurement_mpc['time'], measurement_mpc['PTot'], c=COLORS[1], label='MPC')
plt.plot(measurement_ddqn['time'], measurement_ddqn['PTot'], c=COLORS[2], label='DDQN')
plt.plot(measurement_ppo['time'], measurement_ppo['PTot'], c=COLORS[3], label='PPO')
plt.plot(measurement_sac['time'], measurement_sac['PTot'],c=COLORS[5], label='SAC')
plt.grid(True)
plt.xticks(xticks,xticks_label)
plt.legend(fancybox=True, framealpha=0.5)
plt.ylabel('Total [W]')
plt.xlabel('Time [h]')
plt.savefig('control-response.pdf')
plt.savefig('control-response.png')

"""
# save baseline and mpc measurements from simulation
## save interpolated measurement data for comparison
def interpolate_dataframe(df,new_index):
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name

    for col_name, col in df.items():
        df_out[col_name] = np.interp(new_index, df.index, col)    
    return df_out

measurement_base = pd.DataFrame(measurement_base,index=measurement_base['time'])
measurement_mpc = pd.DataFrame(measurement_mpc,index=measurement_mpc['time'])
measurement_ddqn = pd.DataFrame(measurement_ddqn,index=measurement_ddqn['time'])
measurement_sac = pd.DataFrame(measurement_sac,index=measurement_sac['time'])

tim_intp = np.arange(ts,te,dt)
measurement_base = interpolate_dataframe(measurement_base[['PTot','TRoo']],tim_intp)
measurement_mpc = interpolate_dataframe(measurement_mpc[['PTot','TRoo']],tim_intp)
measurement_ddqn = interpolate_dataframe(measurement_ddqn[['PTot','TRoo']],tim_intp)
measurement_sac = interpolate_dataframe(measurement_sac[['PTot','TRoo']],tim_intp)

#measurement_base.to_csv('measurement_base.csv')
#measurement_mpc.to_csv('measurement_mpc.csv')
#measurement_ddqn.to_csv('measurement_ddqn.csv')
#measurement_sac.to_csv('measurement_sac.csv')

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
    res = penalty * 500.0 + cost*5e4
    
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
#plt.plot(rewards_moving_sac_v1.values,'y--',lw=1,label='SAC')
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
rewards_dqn_v1_last = get_rewards(measurement_ddqn['PTot'].values,measurement_ddqn['TRoo'].values,price_tou,alpha)
rewards_dqn_v1_last = pd.DataFrame(rewards_dqn_v1_last,columns=[['ene_cost','penalty','rewards']])

rewards_sac_v1_last = get_rewards(measurement_sac['PTot'].values,measurement_sac['TRoo'].values,price_tou,alpha)
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

"""
