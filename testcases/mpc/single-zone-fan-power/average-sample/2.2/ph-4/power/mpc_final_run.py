import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import json
# load testbed
from pyfmi import load_fmu

# simulation setup
startTime = 212*24*3600.+13*24*3600.
endTime = startTime + 1*24*3600.
dt = 15*60.

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
res_base = baseline.simulate(start_time = startTime,
                    final_time = endTime, 
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
options['ncp'] = 500.
options['initialize'] = True

i = 0
res_mpc=[]
ts = startTime
while ts < endTime:
  te = ts + dt
  mpc.set(list(["uFan"]),list([u_opt[i]]))
  res = mpc.simulate(start_time = ts,
                      final_time = te, 
                      options = options)
  res_mpc.append(res)

  ts = te
  i += 1
  options['initialize'] = False
################################################################
##           Compare MPC with Baseline
## =============================================================

# read measurements
measurement_names = ['time','TRoo','TOut','PTot','hvac.uFan','hvac.fanSup.m_flow_in', 'senTSetRooCoo.y', 'CO2Roo']
measurement_mpc = {}
measurement_base = {}

for name in measurement_names:
    measurement_base[name] = res_base[name]
    values = []
    for res in res_mpc:
      value = list(res[name])
      values += value
      measurement_mpc[name] = values

## simulate baseline
occ_start = 7
occ_end = 20
tim = np.arange(startTime,endTime,dt)
T_upper = np.array([30.0 for i in tim])
T_upper[occ_start*4:(occ_end-1)*4] = 26.0
T_lower = np.array([18.0 for i in tim])
T_lower[occ_start*4:(occ_end-1)*4] = 22.0


price_tou = [0.0640, 0.0640, 0.0640, 0.0640, 
        0.0640, 0.0640, 0.0640, 0.0640, 
        0.1391, 0.1391, 0.1391, 0.1391, 
        0.3548, 0.3548, 0.3548, 0.3548, 
        0.3548, 0.3548, 0.1391, 0.1391, 
        0.1391, 0.1391, 0.1391, 0.0640]

xticks=np.arange(ts,te,4*3600)
xticks_label = np.arange(0,24,4)

plt.figure()
plt.subplot(411)
plt.step(np.arange(startTime, endTime, 3600.),price_tou, where='post')
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
plt.plot(measurement_base['time'], np.array(measurement_base['TRoo'])-273.15,'b--',label='Baseline')
plt.plot(measurement_mpc['time'], np.array(measurement_mpc['TRoo'])-273.15,'r-',label='MPC')
plt.plot(measurement_base['time'], np.array(measurement_base['senTSetRooCoo.y'])-273.15,'k:',label='Setpoint')
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




