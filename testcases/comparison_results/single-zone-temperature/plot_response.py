import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import json
# load testbed
from pyfmi import load_fmu

# simulation setup
ts = 212*24*3600.#+13*24*3600
nday = 7
period = nday*24*3600.
te = ts + period
dt = 15*60.
nsteps_h = int(3600//dt)
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

## construct optimal input for fmu
res_base = baseline.simulate(start_time = ts,
                    final_time = te, 
                    options = options)

################################################
##           MPC Final Simulation
## =============================================

# read optimal control inputs
with open('u_opt.json') as f:
  opt = json.load(f)

t_opt = opt['t_opt']
u_opt = opt['u_opt']

print len(t_opt)
print len(u_opt)

### 1- Load virtual building model
hvac = load_fmu('SingleZoneTemperature.fmu')

## fmu settings
options = hvac.simulate_options()
options['ncp'] = 500.
options['initialize'] = True
options['result_handling'] = 'memory'
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
##              DRL final run
##===========================================================
# get actions from the last epoch
actions= np.load('history_Action.npy')
u_opt = np.array(actions[-1,:,0])
print u_opt

u_opt = u_opt*0.5 + 273.15 + 12

## fmu settings
hvac.reset()
options = hvac.simulate_options()
options['ncp'] = 500.
options['initialize'] = True
options['result_handling'] = 'memory'
res_drl = []

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
    res_drl.append(ires)

    t += dt 
    i += 1
    options['initialize'] = False


################################################################
##           Compare MPC/DRL with Baseline
## =============================================================

# read measurements
measurement_names = ['time','TRoo','TOut','PCoo.y','hvac.uFan','hvac.fanSup.m_flow_in', 'senTSetRooCoo.y', 'CO2Roo']
measurement_mpc = {}
measurement_base = {}
measurement_drl = {}

for name in measurement_names:
    measurement_base[name] = res_base[name]
    # get mpc results
    value_name_mpc=[]
    for ires in res_mpc:
      value_name_mpc += list(ires[name])
    measurement_mpc[name] = np.array(value_name_mpc)
    # get drl results
    value_name_drl=[]
    for ires in res_drl:
      value_name_drl += list(ires[name])
    measurement_drl[name] = np.array(value_name_drl)

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
plt.plot(measurement_base['time'], measurement_base['senTSetRooCoo.y']-273.15,'b-',label='Baseline')
plt.plot(measurement_mpc['time'], measurement_mpc['senTSetRooCoo.y']-273.15,'b--',label='MPC')
plt.plot(measurement_drl['time'], measurement_drl['senTSetRooCoo.y']-273.15,'r--',label='DRL')
plt.grid(True)
plt.xticks(xticks,[])
plt.legend()
plt.ylabel('Fan Speed')

plt.subplot(413)
plt.plot(measurement_base['time'], measurement_base['TRoo']-273.15,'b-',label='Baseline')
plt.plot(measurement_mpc['time'],  measurement_mpc['TRoo']-273.15,'b--',label='MPC')
plt.plot(measurement_drl['time'],  measurement_drl['TRoo']-273.15,'r--',label='DRL')
plt.plot(tim,T_upper, 'g-.', lw=1,label='Bounds')
plt.plot(tim,T_lower, 'g-.', lw=1)
plt.grid(True)
plt.xticks(xticks,[])
plt.legend()
plt.ylabel('Room Temperature [C]')

plt.subplot(414)
plt.plot(measurement_base['time'], measurement_base['PCoo.y'],'b-',label='Baseline')
plt.plot(measurement_mpc['time'], measurement_mpc['PCoo.y'],'b--',label='MPC')
plt.plot(measurement_drl['time'], measurement_drl['PCoo.y'],'r--',label='DRL')
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

tim_intp = np.arange(ts,te,dt)
measurement_base_intp = interpolate_dataframe(measurement_base[['PCoo.y','TRoo']],tim_intp)
measurement_mpc_intp = interpolate_dataframe(measurement_mpc[['PCoo.y','TRoo']],tim_intp)


measurement_base_intp.to_csv('measurement_base.csv')
measurement_mpc_intp.to_csv('measurement_mpc.csv')