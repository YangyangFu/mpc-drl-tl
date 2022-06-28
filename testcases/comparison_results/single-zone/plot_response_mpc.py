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
##           MPC PH=4
## =============================================

# read optimal control inputs
with open('./mpc/R2/PH=4/u_opt.json') as f:
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
res_mpc4 = []
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
    res_mpc4.append(ires)

    t += dt 
    i += 1
    options['initialize'] = False

################################################
##           MPC PH=8
## =============================================

# read optimal control inputs
with open('./mpc/R2/PH=8/u_opt.json') as f:
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
res_mpc8 = []
hvac.set("zon.roo.T_start", 273.15+25)
# main loop - do step
t = ts
i = 0
while t < te:
    u = u_opt[i]
    hvac.set('uFan', u[0])
    ires = hvac.simulate(start_time=t,
                         final_time=t+dt,
                         options=options)
    res_mpc8.append(ires)

    t += dt
    i += 1
    options['initialize'] = False

################################################
##           MPC PH=16
## =============================================

# read optimal control inputs
with open('./mpc/R2/PH=16/u_opt.json') as f:
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
res_mpc16 = []
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
    res_mpc16.append(ires)

    t += dt 
    i += 1
    options['initialize'] = False

################################################
##           MPC PH=32
## =============================================

# read optimal control inputs
with open('./mpc/R2/PH=32/u_opt.json') as f:
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
res_mpc32 = []
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
    res_mpc32.append(ires)

    t += dt 
    i += 1
    options['initialize'] = False

################################################
##           MPC PH=48
## =============================================

# read optimal control inputs
with open('./mpc/R2/PH=48/u_opt.json') as f:
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
res_mpc48 = []
hvac.set("zon.roo.T_start", 273.15+25)
# main loop - do step
t = ts
i = 0
while t < te:
    u = u_opt[i]
    hvac.set('uFan', u[0])
    ires = hvac.simulate(start_time=t,
                         final_time=t+dt,
                         options=options)
    res_mpc48.append(ires)

    t += dt
    i += 1
    options['initialize'] = False

################################################
##           MPC PH=96
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
res_mpc96 = []
hvac.set("zon.roo.T_start", 273.15+25)
# main loop - do step
t = ts
i = 0
while t < te:
    u = u_opt[i]
    hvac.set('uFan', u[0])
    ires = hvac.simulate(start_time=t,
                         final_time=t+dt,
                         options=options)
    res_mpc96.append(ires)

    t += dt
    i += 1
    options['initialize'] = False

################################################################
##           Compare MPC/DRL with Baseline
## =============================================================

# read measurements
measurement_mpc4 = {}
measurement_mpc8 = {}
measurement_mpc16 = {}
measurement_mpc32 = {}
measurement_mpc48 = {}
measurement_mpc96 = {}
measurement_base = {}

for name in measurement_names:
    measurement_base[name] = res_base[name]
    # get mpc results
    value_name_mpc4=[]
    for ires in res_mpc4:
      value_name_mpc4 += list(ires[name])
    measurement_mpc4[name] = np.array(value_name_mpc4)
    value_name_mpc8=[]
    for ires in res_mpc8:
      value_name_mpc8 += list(ires[name])
    measurement_mpc8[name] = np.array(value_name_mpc8)
    value_name_mpc16=[]
    for ires in res_mpc16:
      value_name_mpc16 += list(ires[name])
    measurement_mpc16[name] = np.array(value_name_mpc16)
    value_name_mpc32 = []
    for ires in res_mpc32:
      value_name_mpc32 += list(ires[name])
    measurement_mpc32[name] = np.array(value_name_mpc32)
    value_name_mpc48 = []
    for ires in res_mpc48:
      value_name_mpc48 += list(ires[name])
    measurement_mpc48[name] = np.array(value_name_mpc48)
    value_name_mpc96 = []
    for ires in res_mpc96:
      value_name_mpc96 += list(ires[name])
    measurement_mpc96[name] = np.array(value_name_mpc96)

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
ax1 = plt.subplot(411)
ax1.step(np.arange(ts, te, 3600.),price_tou, where='post',c='k', label = "Price ($/kWh)")
ax2 = ax1.twinx()
ax2.plot(measurement_base['time'], measurement_base['TOut']-273.15, label='Outdoor Temperature ($^\circ$C)')
ax1.grid(True)
ax1.legend(fancybox=True, framealpha=0.3, loc=2)
ax2.legend(fancybox=True, framealpha=0.3, loc=1)
plt.xticks(xticks,[])

#ax1.set_ylabel('Price ($/kW)')
#ax2.set_ylabel('Outdoor Temperature ($^\circ$C)')

plt.subplot(412)
plt.plot(measurement_base['time'], measurement_base['fcu.uFan'], c=COLORS[0], label='RBC')
#plt.plot(measurement_mpc4['time'], measurement_mpc4['fcu.uFan'],c=COLORS[1], label='MPC:PH=4')
#plt.plot(measurement_mpc8['time'], measurement_mpc8['fcu.uFan'],c=COLORS[2], label='MPC:PH=8')
plt.plot(measurement_mpc16['time'], measurement_mpc16['fcu.uFan'],c=COLORS[1], label='MPC:PH=16')
plt.plot(measurement_mpc32['time'], measurement_mpc32['fcu.uFan'],c=COLORS[2], label='MPC:PH=32')
plt.plot(measurement_mpc48['time'], measurement_mpc48['fcu.uFan'],c=COLORS[3], label='MPC:PH=48')
plt.plot(measurement_mpc96['time'], measurement_mpc96['fcu.uFan'],c=COLORS[4], label='MPC:PH=96')
plt.grid(True)
plt.xticks(xticks,[])
plt.legend(fancybox=True, framealpha=0.3, loc=1)
plt.ylabel('Fan Speed')

plt.subplot(413)
plt.plot(measurement_base['time'], measurement_base['TRoo']-273.15,c=COLORS[0], label='RBC')
#plt.plot(measurement_mpc4['time'], measurement_mpc4['TRoo']-273.15,c=COLORS[1], label='MPC:PH=4')
#plt.plot(measurement_mpc8['time'], measurement_mpc8['TRoo']-273.15,c=COLORS[2], label='MPC:PH=8')
plt.plot(measurement_mpc16['time'], measurement_mpc16['TRoo']-273.15,c=COLORS[1], label='MPC:PH=16')
plt.plot(measurement_mpc32['time'], measurement_mpc32['TRoo']-273.15,c=COLORS[2], label='MPC:PH=32')
plt.plot(measurement_mpc48['time'], measurement_mpc48['TRoo']-273.15,c=COLORS[3], label='MPC:PH=48')
plt.plot(measurement_mpc96['time'], measurement_mpc96['TRoo']-273.15,c=COLORS[4], label='MPC:PH=96')
plt.plot(tim,T_upper, 'k-.', lw=1,label='Bounds')
plt.plot(tim,T_lower, 'k-.', lw=1)
plt.grid(True)
plt.xticks(xticks,[])
plt.legend(fancybox=True, framealpha=0.3, loc=1)
plt.ylabel('Room Temperature ($^\circ$C)')

plt.subplot(414)
plt.plot(measurement_base['time'], measurement_base['PTot'], c=COLORS[0], label='RBC')
#plt.plot(measurement_mpc4['time'], measurement_mpc4['PTot'],c=COLORS[1], label='MPC:PH=4')
#plt.plot(measurement_mpc8['time'], measurement_mpc8['PTot'],c=COLORS[2], label='MPC:PH=8')
plt.plot(measurement_mpc16['time'], measurement_mpc16['PTot'], c=COLORS[1], label='MPC:PH=16')
plt.plot(measurement_mpc32['time'], measurement_mpc32['PTot'],c=COLORS[2], label='MPC:PH=32')
plt.plot(measurement_mpc48['time'], measurement_mpc48['PTot'],c=COLORS[3], label='MPC:PH=48')
plt.plot(measurement_mpc96['time'], measurement_mpc96['PTot'],c=COLORS[4], label='MPC:PH=96')
plt.grid(True)
plt.xticks(xticks,xticks_label)
plt.legend(fancybox=True, framealpha=0.3, loc=1)
plt.ylabel('Power (W)')
plt.xlabel('Time (h)')
plt.savefig('control-response-mpc.pdf')
plt.savefig('control-response-mpc.png')
