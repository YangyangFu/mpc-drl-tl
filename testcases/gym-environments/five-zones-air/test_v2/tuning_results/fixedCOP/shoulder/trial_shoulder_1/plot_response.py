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
ts = 159*24*3600.#+13*24*3600
nday = 7
period = nday*24*3600.
te = ts + period
dt = 15*60.
nsteps_h = int(3600//dt)

# setup for DRL test
alpha = 1
nepochs = 100
high = 1. #action space upper bound
low = -1. #action space lower bound

# define some filters to save simulation time using fmu
measurement_names_base = ['time','TAirDev.y','TOut.y',
'eleTot.y','fanSup.y','yDamMax.y','yDamMin.y',
'conAHU.TSup','conAHU.TSupSet','conAHU.supFan.conSpe.u_s','conAHU.ducStaPre',
'flo.temAirEas.T','flo.temAirSou.T','flo.temAirWes.T','flo.temAirNor.T','flo.temAirPer5.T']

measurement_names_fmu = ['time','TZoneAirDev_y','TOutAir_y',
'GHI_y','PHVAC_y','yFanSpe_y','yDamMax_y','yDamMin_y',
'oveAct_TSupSet','oveAct_dpSet','modCoo.conAHU.TSup','modCoo.conAHU.ducStaPre',
'modCoo.flo.temAirEas.T','modCoo.flo.temAirSou.T','modCoo.flo.temAirWes.T','modCoo.flo.temAirNor.T','modCoo.flo.temAirPer5.T']

##########################################
##          Baseline Simulation
## =========================================
# DEFINE MODEL
# ------------
baseline = load_fmu('FiveZoneAirBaseline.fmu')

## fmu settings
options = baseline.simulate_options()
options['ncp'] = 100
options['initialize'] = True
options['result_handling'] = 'memory'
options['filter'] = measurement_names_base

## construct optimal input for fmu
res_base = baseline.simulate(start_time = ts,
                    final_time = te, 
                    options = options)

print ("Finish baseline simulation")

###############################################################
##              DRL final run: Discrete: v0-ppo
##===========================================================
# get actions from the last epoch
v0_ppo_case = './ppo_results'
actions= np.load(v0_ppo_case+'/his_act_final.npy')
actions = np.clip(actions, -1.0, 1.0)
actions = low + (high - low) * (actions + 1.0) / 2.0
u_opt=273.15+12+6*(actions[:,0]+1)/2
v_opt=25+(410-25)*(actions[:,1]+1)/2

print ("u_opt",u_opt)
print ("v_opt",v_opt)

### 1- Load virtual building model
hvac = load_fmu('FiveZoneAir.fmu')

input_names = hvac.get_model_variables(causality = 2).keys()
print(input_names)
print('Inputs: {0}'.format(input_names))

## fmu settings
options = hvac.simulate_options()
options['ncp'] = 100
options['initialize'] = True
options['result_handling'] = 'memory'
options['filter'] = measurement_names_fmu
res_ppo_v0 = []

## construct optimal input for fmu
# main loop - do step
t = ts
i = 0
while t < te:
    inputdata = np.transpose(np.vstack(([ts,ts+dt],[u_opt[i],u_opt[i]],[v_opt[i],v_opt[i]])))
    #print ("inputdata", inputdata)
    input_object = (list(input_names),inputdata)
    ires = hvac.simulate(start_time = t,
                final_time = t+dt, 
                options = options,
                input = input_object)
    res_ppo_v0.append(ires)
    t += dt 
    i += 1
    options['initialize'] = False


################################################
##           OBC Final Simulation
## =============================================

# # read optimal control inputs
# from buildingspy.io.outputfile import Reader
# opt = "merge.mat"
# r=Reader(opt, "dymola")
# (t, u) = r.values('conAHU.TSupSet')

# t_opt = np.arange(ts,te,dt)
# u_opt = np.interp(t_opt, t, u)

# print(len(u_opt))
# print(u_opt)

# ### 1- Load virtual building model
# #hvac = load_fmu('FiveZoneAir.fmu')

# hvac.reset()
# ## fmu settings
# options = hvac.simulate_options()
# options['ncp'] = 100
# options['initialize'] = True
# options['result_handling'] = 'memory'
# options['filter'] = measurement_names
# res_obc = []

# # main loop - do step
# t = ts
# i = 0
# while t < te:
    # u = u_opt[i]
    # u_traj = np.transpose(np.vstack(([t,t+dt],[u,u])))
    # input_object = ("uTSupSet",u_traj)
    # ires = hvac.simulate(start_time = t,
                # final_time = t+dt, 
                # options = options,
                # input = input_object)
    # res_obc.append(ires)
    # t += dt 
    # i += 1
    # options['initialize'] = False



##############################################################
#           Compare DRL with Baseline
# =============================================================

# read measurements
measurement_base = {}
measurement_ppo_v0 = {}
# measurement_obc = {}

for name in measurement_names_base:
    measurement_base[name] = res_base[name]
    # get ppo_v0 results
for name in measurement_names_fmu:
    value_name_ppo_v0=[]
    for ires in res_ppo_v0:
      value_name_ppo_v0 += list(ires[name])
    measurement_ppo_v0[name] = np.array(value_name_ppo_v0)
    # # get obc results
    # value_name_obc=[]
    # for ires in res_obc:
      # value_name_obc += list(ires[name])
    # measurement_obc[name] = np.array(value_name_obc)    

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

plt.subplot(811)
plt.plot(measurement_base['time'], measurement_base['conAHU.TSup']-273.15,'b--',label='Baseline')
plt.plot(measurement_base['time'], measurement_base['conAHU.TSupSet']-273.15,'b-')
plt.plot(measurement_ppo_v0['time'], measurement_ppo_v0['modCoo.conAHU.TSup']-273.15,'r--',label='ppo')
#plt.plot(measurement_ppo_v0['time'], measurement_ppo_v0['oveAct_TSupSet']-273.15,'r-')
# plt.plot(measurement_obc['time'], measurement_obc['conAHU.TSup']-273.15,'g--',label='OBC')
# plt.plot(measurement_obc['time'], measurement_obc['conAHU.TSupSet']-273.15,'g-')
plt.grid(True)
plt.xticks(xticks,[])
plt.legend()
plt.ylabel('Supply air temperature [C]')

plt.subplot(812)
plt.plot(measurement_base['time'], measurement_base['conAHU.ducStaPre'],'b--',label='Baseline')
plt.plot(measurement_base['time'], measurement_base['conAHU.supFan.conSpe.u_s']*410,'b-')
plt.plot(measurement_ppo_v0['time'], measurement_ppo_v0['modCoo.conAHU.ducStaPre'],'r--',label='ppo')
#plt.plot(measurement_ppo_v0['time'], measurement_ppo_v0['oveAct_dpSet'],'r-')

plt.grid(True)
plt.xticks(xticks,[])
plt.legend()
plt.ylabel('Supply air DP [C]')


plt.subplot(813)
plt.plot(measurement_base['time'], measurement_base['flo.temAirSou.T']-273.15,'b-',label='Baseline')
plt.plot(measurement_ppo_v0['time'],  measurement_ppo_v0['modCoo.flo.temAirSou.T']-273.15,'r--',label='ppo')
# plt.plot(measurement_obc['time'],  measurement_obc['modCoo.flo.temAirSou.T']-273.15,'g-.',label='OBC')
plt.plot(tim,T_upper, 'g-.', lw=1,label='Bounds')
plt.plot(tim,T_lower, 'g-.', lw=1)
plt.grid(True)
plt.xticks(xticks,[])
plt.legend()
plt.ylabel('South Zone Temperature [C]')
plt.ylim(22, 26)

plt.subplot(814)
plt.plot(measurement_base['time'], measurement_base['flo.temAirEas.T']-273.15,'b-',label='Baseline')
plt.plot(measurement_ppo_v0['time'],  measurement_ppo_v0['modCoo.flo.temAirEas.T']-273.15,'r--',label='ppo')
# plt.plot(measurement_obc['time'],  measurement_obc['modCoo.flo.temAirEas.T']-273.15,'g-.',label='OBC')
plt.plot(tim,T_upper, 'g-.', lw=1,label='Bounds')
plt.plot(tim,T_lower, 'g-.', lw=1)
plt.grid(True)
plt.xticks(xticks,[])
plt.legend()
plt.ylabel('East Zone Temperature [C]')
plt.ylim(22, 26)  

plt.subplot(815)
plt.plot(measurement_base['time'], measurement_base['flo.temAirWes.T']-273.15,'b-',label='Baseline')
plt.plot(measurement_ppo_v0['time'],  measurement_ppo_v0['modCoo.flo.temAirWes.T']-273.15,'r--',label='ppo')
# plt.plot(measurement_obc['time'],  measurement_obc['modCoo.flo.temAirWes.T']-273.15,'g-.',label='OBC')
plt.plot(tim,T_upper, 'g-.', lw=1,label='Bounds')
plt.plot(tim,T_lower, 'g-.', lw=1)
plt.grid(True)
plt.xticks(xticks,[])
plt.legend()
plt.ylabel('West Zone Temperature [C]')
plt.ylim(22, 26)  

plt.subplot(816)
plt.plot(measurement_base['time'], measurement_base['flo.temAirNor.T']-273.15,'b-',label='Baseline')
plt.plot(measurement_ppo_v0['time'],  measurement_ppo_v0['modCoo.flo.temAirNor.T']-273.15,'r--',label='ppo')
# plt.plot(measurement_obc['time'],  measurement_obc['modCoo.flo.temAirNor.T']-273.15,'g-.',label='OBC')
plt.plot(tim,T_upper, 'g-.', lw=1,label='Bounds')
plt.plot(tim,T_lower, 'g-.', lw=1)
plt.grid(True)
plt.xticks(xticks,[])
plt.legend()
plt.ylabel('North Zone Temperature [C]')
plt.ylim(22, 26)  

plt.subplot(817)
plt.plot(measurement_base['time'], measurement_base['flo.temAirPer5.T']-273.15,'b-',label='Baseline')
plt.plot(measurement_ppo_v0['time'],  measurement_ppo_v0['modCoo.flo.temAirPer5.T']-273.15,'r--',label='ppo')
# plt.plot(measurement_obc['time'],  measurement_obc['modCoo.flo.temAirPer5.T']-273.15,'g-.',label='OBC')
plt.plot(tim,T_upper, 'g-.', lw=1,label='Bounds')
plt.plot(tim,T_lower, 'g-.', lw=1)
plt.grid(True)
plt.xticks(xticks,[])
plt.legend()
plt.ylabel('Core Zone Temperature [C]')
plt.ylim(22, 26)  

plt.subplot(818)
plt.plot(measurement_base['time'], measurement_base['eleTot.y'],'b-',label='Baseline')
plt.plot(measurement_ppo_v0['time'], measurement_ppo_v0['PHVAC_y'],'r--',label='ppo')
# plt.plot(measurement_obc['time'],  measurement_obc['PHVAC_y'],'g-.',label='OBC')
plt.grid(True)
plt.xticks(xticks,xticks_label)
plt.ylabel('Total [W]')
plt.legend()
plt.savefig('drl_comparison.pdf')


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
measurement_ppo_v0 = pd.DataFrame(measurement_ppo_v0,index=measurement_ppo_v0['time'])
# measurement_obc = pd.DataFrame(measurement_obc,index=measurement_obc['time'])

tim_intp = np.arange(ts,te,dt)
measurement_base = interpolate_dataframe(measurement_base[['eleTot.y','TAirDev.y','flo.temAirEas.T','flo.temAirSou.T','flo.temAirWes.T','flo.temAirNor.T','flo.temAirPer5.T']],tim_intp)
measurement_ppo_v0 = interpolate_dataframe(measurement_ppo_v0[['PHVAC_y','TZoneAirDev_y','modCoo.flo.temAirEas.T','modCoo.flo.temAirSou.T','modCoo.flo.temAirWes.T','modCoo.flo.temAirNor.T','modCoo.flo.temAirPer5.T']],tim_intp)
# measurement_obc = interpolate_dataframe(measurement_obc[['PHVAC','TAirDev.y','TRooAirSou','TRooAirEas','TRooAirNor','TRooAirWes','TRooAirCor']],tim_intp)
#measurement_base.to_csv('measurement_base.csv')
#measurement_ppo_v0.to_csv('measurement_ppo_v0.csv')
#measurement_obc.to_csv('measurement_obc.csv')

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
    res = (penalty * 5000 + cost*500) / 1000
    return res
    
def get_rewards(PHVAC,TZoneVio,price_tou,alpha):
    n= len(PHVAC)
    energy_cost = []
    penalty = []
    rewards = []
    for i in range(n):
        # assume 1 step is 15 minutes and data starts from hour 0
        hindex = (i%(nsteps_h*24))//nsteps_h
        power=PHVAC[i]
        price = price_tou[hindex]
        # the power should divide by 1000
        energy_cost.append(power/1000./nsteps_h*price)
        # zone temperature penalty
        if 7 <= hindex <= 19:
            penalty.append(alpha*TZoneVio[i])
        else:
            penalty.append(0)
        rewards.append(rw_func(energy_cost[-1], penalty[-1]))
    return -np.array([energy_cost, penalty, rewards]).transpose()

#### get rewards
#================================================================================
rewards_base = get_rewards(measurement_base['eleTot.y'].values,measurement_base['TAirDev.y'].values,price_tou,alpha)
rewards_base = pd.DataFrame(rewards_base,columns=[['ene_cost','penalty','rewards']])

# rewards_obc = get_rewards(measurement_obc['PHVAC'].values,measurement_obc['TAirDev.y'].values,price_tou,alpha)
# rewards_obc = pd.DataFrame(rewards_obc,columns=[['ene_cost','penalty','rewards']])


# get rewards - DRL - we can either read from training results or 
# recalculate using the method for mpc and baseline (very time consuming for multi-epoch training)
rewards_ppo_v0_hist = pd.read_csv('run-.-tag-train_reward.csv')
rewards_ppo_v0_hist.iloc[:,1] = rewards_ppo_v0_hist.iloc[:,1] / rewards_ppo_v0_hist.iloc[0,1]
new_index=np.linspace(1, nepochs, num=100, endpoint=True)
rewards_ppo_v0 = np.interp(new_index,rewards_ppo_v0_hist.iloc[:,1], rewards_ppo_v0_hist.iloc[:,2])




# plot rewards with moving windows - epoch-by-epoch
moving = nday*24*3600.//dt 
rewards_moving_base = rewards_base['rewards'].groupby(rewards_base.index//moving).sum()
# rewards_moving_obc = rewards_obc['rewards'].groupby(rewards_obc.index//moving).sum()


plt.figure(figsize=(9,6))
plt.plot(list(rewards_moving_base.values)*nepochs,'b-',label='RBC')
plt.plot(rewards_ppo_v0,'r--',lw=1,label='ppo')
# plt.plot(list(rewards_moving_obc.values)*nepochs,'g-',label='OBC')

plt.ylabel('rewards')
plt.xlabel('epoch')
plt.grid(True)
plt.legend()
plt.savefig('rewards_epoch.pdf')
#plt.savefig('rewards_epoch.png')


# The following codes are only for comparing occupied control performance
#===========================================================================
#flag = np.logical_or(rewards_ppo_v0_last.index%96//4<7,rewards_ppo_v0_last.index%96//4>=19)
#rewards_ppo_v0_last.loc[flag,'ene_cost'] = 0
#print rewards_ppo_v0_last['ene_cost'].sum()

# save total energy cost, violations etc using the final episode for DRL
rewards_ppo_v0_last = get_rewards(measurement_ppo_v0['PHVAC_y'].values,measurement_ppo_v0['TZoneAirDev_y'].values,price_tou,alpha)
rewards_ppo_v0_last = pd.DataFrame(rewards_ppo_v0_last,columns=[['ene_cost','penalty','rewards']])

# # save total energy cost, violations etc using the final episode for OBC
# rewards_obc_last = get_rewards(measurement_obc['PHVAC'].values,measurement_obc['TAirDev.y'].values,price_tou,alpha)
# rewards_obc_last = pd.DataFrame(rewards_obc_last,columns=[['ene_cost','penalty','rewards']])


comparison={'base':{'energy_cost':list(rewards_base['ene_cost'].sum()),
                    'temp_violation':list(rewards_base['penalty'].sum())},
            'ppo':{'energy_cost':list(rewards_ppo_v0_last['ene_cost'].sum()),
                    'temp_violation':list(rewards_ppo_v0_last['penalty'].sum())}}#,
            # 'obc':{'energy_cost':list(rewards_obc_last['ene_cost'].sum()),
                    # 'temp_violation':list(rewards_obc_last['penalty'].sum())}                    
                    # }
with open('comparison_epoch.json', 'w') as outfile:
    json.dump(comparison, outfile)

