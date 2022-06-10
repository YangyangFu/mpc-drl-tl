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
options['ncp'] = 672
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
    hvac.set('uFan',u[0])
    ires = hvac.simulate(start_time = t,
                final_time = t+dt, 
                options = options)
    res_mpc.append(ires)

    t += dt 
    i += 1
    options['initialize'] = False


###############################################################
##              DRL final run: ddqn-seed5
##===========================================================
# get actions from the last epoch
ddqn_case = './DRL-R2/ddqn_seed5/'
with open(ddqn_case+'u_opt.json') as f:
  u_opt = json.load(f)

## fmu settings
hvac.reset()
options = hvac.simulate_options()
options['ncp'] = 15
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
sac_case = './DRL-R2/sac_seed1/'
with open(sac_case+'u_opt.json') as f:
  u_opt = json.load(f)

## fmu settings
hvac.reset()
options = hvac.simulate_options()
options['ncp'] = 15
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
options['ncp'] = 15
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

###############################################################
##              DRL final run: qrdqn_seed3:
##===========================================================
# get actions from the last epoch
qrdqn_case = './DRL-R2/qrdqn_seed3/'
with open(qrdqn_case+'u_opt.json') as f:
  u_opt = json.load(f)

## fmu settings
hvac.reset()
options = hvac.simulate_options()
options['ncp'] = 15
options['initialize'] = True
options['result_handling'] = 'memory'
options['filter'] = measurement_names
res_qrdqn = []
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
    res_qrdqn.append(ires)

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
measurement_qrdqn = {}

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
    # get qrdqn
    value_name_qrdqn = []
    for ires in res_qrdqn:
      value_name_qrdqn += list(ires[name])
    measurement_qrdqn[name] = np.array(value_name_qrdqn)
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

matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams['lines.linestyle'] = '-'

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
plt.plot(measurement_qrdqn['time'], measurement_qrdqn['fcu.uFan'],c=COLORS[4], label='QRDQN')
plt.plot(measurement_sac['time'], measurement_sac['fcu.uFan'],c=COLORS[5],label='SAC')
plt.grid(True)
plt.xticks(xticks,[])
plt.legend(fancybox=True, framealpha=0.3, loc=1)
plt.ylabel('Fan Speed')

plt.subplot(413)
plt.plot(measurement_base['time'], measurement_base['TRoo']-273.15,c=COLORS[0], label='RBC')
plt.plot(measurement_mpc['time'],  measurement_mpc['TRoo']-273.15,c=COLORS[1], label='MPC')
plt.plot(measurement_ddqn['time'],  measurement_ddqn['TRoo']-273.15,c=COLORS[2], label='DDQN')
plt.plot(measurement_ppo['time'],  measurement_ppo['TRoo']-273.15,c=COLORS[3], label='PPO')
plt.plot(measurement_qrdqn['time'],  measurement_qrdqn['TRoo']-273.15,c=COLORS[4], label='QRDQN')
plt.plot(measurement_sac['time'],  measurement_sac['TRoo']-273.15,c=COLORS[5],label='SAC')
plt.plot(tim,T_upper, 'k-.', lw=1,label='Bounds')
plt.plot(tim,T_lower, 'k-.', lw=1)
plt.grid(True)
plt.xticks(xticks,[])
plt.legend(fancybox=True, framealpha=0.3, loc=1)
plt.ylabel('Room Temperature [C]')

plt.subplot(414)
plt.plot(measurement_base['time'], measurement_base['PTot'], c=COLORS[0], label='RBC')
plt.plot(measurement_mpc['time'], measurement_mpc['PTot'], c=COLORS[1], label='MPC')
plt.plot(measurement_ddqn['time'], measurement_ddqn['PTot'], c=COLORS[2], label='DDQN')
plt.plot(measurement_ppo['time'], measurement_ppo['PTot'], c=COLORS[3], label='PPO')
plt.plot(measurement_qrdqn['time'], measurement_qrdqn['PTot'],c=COLORS[4], label='QRDQN')
plt.plot(measurement_sac['time'], measurement_sac['PTot'],c=COLORS[5], label='SAC')
plt.grid(True)
plt.xticks(xticks,xticks_label)
plt.legend(fancybox=True, framealpha=0.3, loc=1)
plt.ylabel('Total [W]')
plt.xlabel('Time [h]')
plt.savefig('control-response.pdf')
plt.savefig('control-response.png')

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
measurement_ddqn = pd.DataFrame(measurement_ddqn,index=measurement_ddqn['time'])
measurement_ppo = pd.DataFrame(measurement_ppo,index=measurement_ppo['time'])
measurement_qrdqn = pd.DataFrame(measurement_qrdqn,index=measurement_qrdqn['time'])
measurement_sac = pd.DataFrame(measurement_sac,index=measurement_sac['time'])

tim_intp = np.arange(ts,te+1,dt)
measurement_base = interpolate_dataframe(measurement_base[['PTot','TRoo', 'fcu.uFan']],tim_intp)
measurement_mpc = interpolate_dataframe(measurement_mpc[['PTot','TRoo', 'fcu.uFan']],tim_intp)
measurement_ddqn = interpolate_dataframe(measurement_ddqn[['PTot','TRoo', 'fcu.uFan']],tim_intp)
measurement_ppo = interpolate_dataframe(measurement_ppo[['PTot','TRoo', 'fcu.uFan']],tim_intp)
measurement_qrdqn = interpolate_dataframe(measurement_qrdqn[['PTot','TRoo', 'fcu.uFan']],tim_intp)
measurement_sac = interpolate_dataframe(measurement_sac[['PTot','TRoo', 'fcu.uFan']],tim_intp)

#measurement_base = measurement_base.groupby(measurement_base.index//900).mean()
#measurement_mpc = measurement_mpc.groupby(measurement_mpc.index//900).mean()

weights = [100., 1., 10.]
def rw_func(cost, penalty, delta_action):

    res = - weights[0]*cost - weights[1]*penalty*penalty \
        - delta_action*delta_action*weights[2]

    return res

def get_rewards(u, Ptot,TZone,price_tou):
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
        hindex = (i%(nsteps_h*24))//nsteps_h
        power=Ptot[i]
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
            overshoot.append(np.array([float((TZone[i] -273.15) - T_upper[hindex]), 0.0]).max())
            undershoot.append(np.array([float(T_lower[hindex] - (TZone[i]-273.15)), 0.0]).max())

        penalty.append(sum(np.array(overshoot)) + sum(np.array(undershoot)))
    
        # action changes
        delta_action.append(abs(u[i] - u_prev[i]))

        # sum up for rewards
        rewards.append(rw_func(energy_cost[-1], penalty[-1], delta_action[-1]))

    return np.array([energy, energy_cost, penalty, delta_action, rewards]).transpose()

#### get rewards
#================================================================================
rewards_base = get_rewards(measurement_base['fcu.uFan'].values, measurement_base['PTot'].values,measurement_base['TRoo'].values,price_tou)
rewards_mpc = get_rewards(measurement_mpc['fcu.uFan'].values, measurement_mpc['PTot'].values,measurement_mpc['TRoo'].values,price_tou)
rewards_ddqn = get_rewards(measurement_ddqn['fcu.uFan'].values, measurement_ddqn['PTot'].values,measurement_ddqn['TRoo'].values,price_tou)
rewards_ppo = get_rewards(measurement_ppo['fcu.uFan'].values, measurement_ppo['PTot'].values,measurement_ppo['TRoo'].values,price_tou)
rewards_qrdqn = get_rewards(measurement_qrdqn['fcu.uFan'].values, measurement_qrdqn['PTot'].values,measurement_qrdqn['TRoo'].values,price_tou)
rewards_sac = get_rewards(measurement_sac['fcu.uFan'].values, measurement_sac['PTot'].values,measurement_sac['TRoo'].values,price_tou)

rewards_base = pd.DataFrame(rewards_base,columns=[['energy','ene_cost','penalty', 'delta_action', 'rewards']])
rewards_mpc = pd.DataFrame(rewards_mpc,columns=[['energy','ene_cost','penalty','delta_action', 'rewards']])
rewards_ddqn = pd.DataFrame(rewards_ddqn,columns=[['energy','ene_cost','penalty','delta_action', 'rewards']])
rewards_ppo = pd.DataFrame(rewards_ppo,columns=[['energy','ene_cost','penalty','delta_action', 'rewards']])
rewards_qrdqn = pd.DataFrame(rewards_qrdqn,columns=[['energy','ene_cost','penalty','delta_action', 'rewards']])
rewards_sac = pd.DataFrame(rewards_sac,columns=[['energy','ene_cost','penalty','delta_action', 'rewards']])

# get rewards - DRL - we can either read from training results or 
# recalculate using the method for mpc and baseline (very time consuming for multi-epoch training)
### ===============
mpc_drl_kpis = {'base': {'rewards': float(rewards_base['rewards'].sum()),
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
                        'delta_action_sqaured': float((rewards_mpc['delta_action']**2).sum())},
                'ddqn': {'rewards': float(rewards_ddqn['rewards'].sum()),
                         'energy': float(rewards_ddqn['energy'].sum()),
                         'ene_cost': float(rewards_ddqn['ene_cost'].sum()),
                         'total_temp_violation': float(rewards_ddqn['penalty'].sum()/4),
                         'max_temp_violation': float(rewards_ddqn['penalty'].max()),
                         'delta_action': float(rewards_ddqn['delta_action'].sum()/4),
                         'temp_violation_squared': float((rewards_ddqn['penalty']**2).sum()),
                         'delta_action_sqaured': float((rewards_ddqn['delta_action']**2).sum())},
                'ppo': {'rewards': float(rewards_ppo['rewards'].sum()),
                        'energy': float(rewards_ppo['energy'].sum()),
                        'ene_cost': float(rewards_ppo['ene_cost'].sum()),
                        'total_temp_violation': float(rewards_ppo['penalty'].sum()/4),
                        'max_temp_violation': float(rewards_ppo['penalty'].max()),
                        'delta_action': float(rewards_ppo['delta_action'].sum()/4),
                        'temp_violation_squared': float((rewards_ppo['penalty']**2).sum()),
                        'delta_action_sqaured': float((rewards_ppo['delta_action']**2).sum())},
                'qrdqn': {'rewards': float(rewards_qrdqn['rewards'].sum()),
                          'energy': float(rewards_qrdqn['energy'].sum()),
                          'ene_cost': float(rewards_qrdqn['ene_cost'].sum()),
                          'total_temp_violation': float(rewards_qrdqn['penalty'].sum()/4),
                          'max_temp_violation': float(rewards_qrdqn['penalty'].max()),
                          'delta_action': float(rewards_qrdqn['delta_action'].sum()/4),
                          'temp_violation_squared': float((rewards_qrdqn['penalty']**2).sum()),
                          'delta_action_sqaured': float((rewards_qrdqn['delta_action']**2).sum())},
                'sac': {'rewards': float(rewards_sac['rewards'].sum()),
                        'energy': float(rewards_sac['energy'].sum()),
                        'ene_cost': float(rewards_sac['ene_cost'].sum()),
                        'total_temp_violation': float(rewards_sac['penalty'].sum()/4),
                        'max_temp_violation': float(rewards_sac['penalty'].max()),
                        'delta_action': float(rewards_sac['delta_action'].sum()/4),
                        'temp_violation_squared': float((rewards_sac['penalty']**2).sum()),
                        'delta_action_sqaured': float((rewards_sac['delta_action']**2).sum())}
                }

with open('/control-response-kpis.json', 'w') as outfile:
    json.dump(mpc_drl_kpis, outfile)

pd.DataFrame(mpc_drl_kpis).transpose().to_csv('control-response-kpis.csv')


## ===================================================
##       calculate the shifted load
## ====================================================
shift_mpc = (measurement_mpc['PTot'] - measurement_base['PTot']).abs().sum()/4/1000/2 # kwH
shift_ddqn = (measurement_ddqn['PTot'] - measurement_base['PTot']).abs().sum()/4/1000/2 # kwH
shift_qrdqn = (measurement_qrdqn['PTot'] - measurement_base['PTot']).abs().sum()/4/1000/2 # kwH
shift_ppo = (measurement_ppo['PTot'] - measurement_base['PTot']).abs().sum()/4/1000/2 # kwH
shift_sac = (measurement_sac['PTot'] - measurement_base['PTot']).abs().sum()/4/1000/2 # kwH

shift_energy = {'mpc': shift_mpc,
                'ddqn': shift_ddqn,
                'qrdqn': shift_qrdqn,
                'ppo': shift_ppo,
                'sac': shift_sac}

with open("shifted-energy.json", 'w') as file:
    json.dump(shift_energy, file)