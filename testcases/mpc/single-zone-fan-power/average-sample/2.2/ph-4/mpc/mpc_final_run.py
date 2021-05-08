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
baseline = load_fmu('SingleZoneDamperControlBaseline.fmu')

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
mpc = load_fmu('SingleZoneDamperControl.fmu')

## fmu settings
options = mpc.simulate_options()
options['ncp'] = 5000.
options['initialize'] = True

## construct optimal input for fmu
u_traj = np.transpose(np.vstack((t_opt,u_opt)))
input_object = ("uFan",u_traj)
print u_traj
res_mpc = mpc.simulate(start_time = ts,
                    final_time = te, 
                    options = options,
                    input = input_object)

################################################################
##           Compare MPC with Baseline
## =============================================================

# read measurements
measurement_names = ['time','TRoo','TOut','PTot','hvac.uFan','hvac.fanSup.m_flow_in', 'senTSetRooCoo.y', 'CO2Roo']
measurement_mpc = {}
measurement_base = {}

for name in measurement_names:
    measurement_base[name] = res_base[name]
    measurement_mpc[name] = res_mpc[name]

## simulate baseline
occ_start = 7
occ_end = 19
tim = np.arange(ts,te,dt)
T_upper = np.array([30.0 for i in tim])
#T_upper[occ_start*4:(occ_end-1)*4] = 26.0
T_lower = np.array([18.0 for i in tim])
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

xticks=np.arange(ts,te,4*3600)
xticks_label = np.arange(0,24*nday,4)

plt.figure()
plt.subplot(411)
plt.step(np.arange(ts, te, 3600.),price_tou, where='post')
plt.xticks(xticks,[])
plt.ylabel('Price ($/kW)')

plt.subplot(412)
plt.plot(measurement_base['time'], measurement_base['hvac.uFan'],'b--',label='Baseline')
plt.plot(measurement_mpc['time'], measurement_mpc['hvac.uFan'],'r-',label='MPC')
plt.grid(True)
plt.xticks(xticks,[])
plt.legend()
plt.ylabel('Fan Speed')

plt.subplot(413)
plt.plot(measurement_base['time'], measurement_base['TRoo']-273.15,'b--',label='Baseline')
plt.plot(measurement_mpc['time'], measurement_mpc['TRoo']-273.15,'r-',label='MPC')
plt.plot(measurement_base['time'], measurement_base['senTSetRooCoo.y']-273.15,'k:',label='Setpoint')
plt.plot(tim,T_upper, 'g-.', lw=1,label='Bounds')
plt.plot(tim,T_lower, 'g-.', lw=1)
plt.grid(True)
plt.xticks(xticks,[])
plt.legend()
plt.ylabel('Room Temperature [C]')

plt.subplot(414)
plt.plot(measurement_base['time'], measurement_base['PTot'],'b--',label='Baseline')
plt.plot(measurement_mpc['time'], measurement_mpc['PTot'],'r-',label='MPC')
plt.grid(True)
plt.xticks(xticks,xticks_label)
plt.ylabel('Total [W]')
plt.savefig('mpc-vs-rbc.pdf')




