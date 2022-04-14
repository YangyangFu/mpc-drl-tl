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

# simulation setup
ts = 212*24*3600.#+13*24*3600
nday = 1
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
P_pred = opt['power_predicted']
Tz_pred = opt['Tz_predicted']
### 1- Load virtual building model
mpc = load_fmu('SingleZoneTemperature.fmu')

## fmu settings
options = mpc.simulate_options()
options['ncp'] = 500.
options['initialize'] = True

## construct optimal input for fmu
res_mpc=[]
i=0
while i<len(t_opt):
  its=t_opt[i]
  ite=its+dt
  iu = u_opt[i]
  u_traj = np.transpose(np.vstack(([its,ite],[iu,iu])))
  input_object = ("TSetCoo",u_traj)
  ires_mpc = mpc.simulate(start_time = its,
                    final_time = ite, 
                    options = options,
                    input = input_object)
  res_mpc.append(ires_mpc)
  i += 1
  options['initialize'] = False
################################################################
##           Compare MPC with Baseline
## =============================================================

# read measurements
measurement_names = ['time','TRoo','TOut','PCoo.y', 'senTSetRooCoo.y']
measurement_mpc = {}
measurement_base = {}

for name in measurement_names:
    measurement_base[name] = res_base[name]

    value_name_mpc=[]
    for ires in res_mpc:
      value_name_mpc += list(ires[name])
    measurement_mpc[name] = value_name_mpc

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

price_tou = [0.02987, 0.02987, 0.02987, 0.02987, 
        0.02987, 0.02987, 0.04667, 0.04667, 
        0.04667, 0.04667, 0.04667, 0.04667, 
        0.04667, 0.04667, 0.15877, 0.15877, 
        0.15877, 0.15877, 0.15877, 0.04667, 
        0.04667, 0.04667, 0.02987, 0.02987]*nday

def interpolate_dataframe(df,new_index):
    """Interpolate a dataframe along its index based on a new index
    """
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name

    for col_name, col in df.items():
        df_out[col_name] = np.interp(new_index, df.index, col)    
    return df_out

measurement_mpc = pd.DataFrame(measurement_mpc,index=measurement_mpc['time'])
measurement_mpc = interpolate_dataframe(measurement_mpc,tim)
print (measurement_mpc)

xticks=np.arange(ts,te+1,12*3600)
xticks_label = np.arange(0,24*nday+1,12)

plt.figure(figsize=(16,12))
plt.subplot(411)
price_plot = price_tou[:]
price_plot.append(price_plot[0])
plt.step(np.arange(ts, te+1, 3600.), price_plot, where='post')
plt.grid(True)
plt.xticks(xticks,[])
plt.ylabel('Price ($/kW)')

plt.subplot(412)
plt.step(measurement_base['time'], measurement_base['senTSetRooCoo.y']-273.15,'b--',label='Baseline')
plt.step(np.array(t_opt), np.array(u_opt)-273.15,'r-',where='post',label='MPC')
plt.grid(True)
plt.xticks(xticks,[])
plt.legend()
plt.ylabel('Cooling Setpoint')

plt.subplot(413)
plt.plot(measurement_base['time'], measurement_base['TRoo']-273.15,'b--',label='Baseline')
plt.plot(measurement_mpc['time'], np.array(measurement_mpc['TRoo'])-273.15,'r-',label='MPC')
plt.plot(t_opt, np.array(Tz_pred)-273.15,'k-',label='Prediction')
plt.plot(tim,T_upper, 'g-.', lw=1,label='Bounds')
plt.plot(tim,T_lower, 'g-.', lw=1)
plt.grid(True)
plt.xticks(xticks,[])
plt.legend()
plt.ylabel('Room Temperature [C]')

plt.subplot(414)
plt.plot(measurement_base['time'], measurement_base['PCoo.y'],'b--',label='Baseline')
plt.plot(measurement_mpc['time'], measurement_mpc['PCoo.y'],'r-',label='MPC')
plt.plot(t_opt,P_pred,'k-',label='Prediction')
plt.grid(True)
plt.xticks(xticks,xticks_label)
plt.legend()
plt.ylabel('Total [W]')
plt.savefig('mpc-vs-rbc.pdf')
plt.savefig('mpc-vs-rbc.png')




