# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

import numpy as np
import pandas as pd
import json
# load testbed
from pyfmi import load_fmu
# import mpc
from mpc import mpc_case

# get measurement
def get_measurement(fmu_result,names):
    if 'time' not in names:
        names.append('time')
    
    dic = {}
    for name in names:
        dic[name] = fmu_result[name][-1]

    # return a pandas data frame
    return pd.DataFrame(dic,index=[dic['time']])
    
def interpolate_dataframe(df,new_index):
    """Interpolate a dataframe along its index based on a new index
    """
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name

    for col_name, col in df.items():
        df_out[col_name] = np.interp(new_index, df.index, col)    
    return df_out

def LIFO(a_list,x):
    """Last in first out: 
    x: scalor
    """
    a_list.append(x)

    return a_list[1:]

def get_states(states,measurement,Tz_pred):
    # read list
    Tz_his = states['Tz_his_meas']
    To_his = states['To_his_meas']
    Tz_his_pred = states['Tz_his_pred']

    # read scalor: to degC for ARX model
    Tz = measurement['TRoo'].values[0] - 273.15
    To = measurement['TOut'].values[0] - 273.15

    # new list - avoid immutable lists
    new_Tz_his = LIFO(Tz_his,Tz)
    new_To_his = LIFO(To_his,To)
    new_Tz_his_pred = LIFO(Tz_his_pred,Tz_pred)

    # new dic
    states['Tz_his_meas'] = new_Tz_his
    states['To_his_meas'] = new_To_his
    states['Tz_his_pred'] = new_Tz_his_pred

    return states

def get_price(time,dt,PH):
    price_tou = [0.02987, 0.02987, 0.02987, 0.02987, 
        0.02987, 0.02987, 0.04667, 0.04667, 
        0.04667, 0.04667, 0.04667, 0.04667, 
        0.15877, 0.15877, 0.15877, 0.15877,
        0.15877, 0.15877, 0.15877, 0.04667, 
        0.04667, 0.04667, 0.02987, 0.02987]
    t_ph = np.arange(time,time+dt*PH,dt)
    price_ph = [price_tou[int(t % 86400 /3600)] for t in t_ph]

    return price_ph

def read_temperature(weather_file,dt):
    """Read temperature and solar radiance from epw file. 
        This module serves as an ideal weather predictor.
    :return: a data frame at an interval of defined time_step
    """
    from pvlib.iotools import read_epw

    dat = read_epw(weather_file)

    tem_sol_h = dat[0][['temp_air']]
    index_h = np.arange(0,3600.*len(tem_sol_h),3600.)
    tem_sol_h.index = index_h

    # interpolate temperature into simulation steps
    index_step = np.arange(0,3600.*len(tem_sol_h),dt)
    return interpolate_dataframe(tem_sol_h,index_step)

def get_Toa(time,dt,PH,Toa_year):
    index_ph = np.arange(time,time+dt*PH,dt)
    Toa = Toa_year.loc[index_ph,:]

    return list(Toa.values.flatten())

### 0- Simulation setup
start = 201*24*3600. # 181 - 7/1 
end = start + 7*24*3600.

### 1- Load virtual building model
hvac = load_fmu('SingleZoneFCU.fmu')

## fmu settings
options = hvac.simulate_options()
options['ncp'] = 100
options['initialize'] = True

# Warm up FMU simulation settings
ts = start
te_warm = ts + 1*3600

### 2- Initialize MPC case 
dt = 15*60.
PH = 48
CH = 1
with open('zone_arx.json') as f:
  parameters_zone = json.load(f)

with open('power.json') as f:
  parameters_power = json.load(f)

# initialize measurement
measurement_names=['TRoo','TOut','PTot','uFan','m_flow_in']
hvac.set("zon.roo.T_start",273.15+25)
res = hvac.simulate(start_time = ts,
                    final_time = ts,
                    options=options)
options['initialize'] = False
measurement_ini = get_measurement(res,measurement_names)

# read one-year weather file
weather_file = 'USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw'
Toa_year = read_temperature(weather_file,dt)

### ===========================
# states at current time for MPC model - this should be customized based on mpc design
lag_Tz = 4 # 4-step lag - should be identified for MPC model
lag_To = 4 # 4-step lag 
Tz_ini = measurement_ini['TRoo'].values[0] - 273.15
P_ini = 0.0
To_his_meas_init = get_Toa(ts-(lag_To-1)*dt,dt,lag_To,Toa_year)

states_ini = {'Tz_his_meas':[Tz_ini]*lag_Tz,
            'To_his_meas':To_his_meas_init,
            'Tz_his_pred':[Tz_ini]*lag_Tz} # initial states used for MPC models

### ==========================================
### predictors
predictor = {}
# energy prices 
predictor['price'] = get_price(ts,dt,PH)
# outdoor air temperature
predictor['Toa'] = get_Toa(ts,dt,PH,Toa_year)

### ==================================
### 3- MPC Control Loop
mFan_nominal=0.55 # kg/s
uFan_ini = 0.
# initialize fan speed for warmup setup
uFan = uFan_ini
states = states_ini
measurement = measurement_ini
# initialize mpc case 
case = mpc_case(PH=PH,
                CH=CH,
                time=ts,
                dt=dt,
                zone_model = parameters_zone,
                power_model = parameters_power,
                measurement = measurement,
                states = states,
                predictor = predictor)
# initialize all results
u_opt=[]
t_opt=[]
P_pred_opt = []
Tz_pred_opt = []
warmup = True

while ts<end:
    
    te = ts+dt*CH
    t_opt.append(ts)

    ### generate control action from MPC
    print("\nstate 1")
    print(case.states) 
    if not warmup: # activate mpc after warmup
        # update mpc case
        case.set_time(ts)
        case.set_measurement(measurement)
        case.set_states(states)   
        case.set_predictor(predictor)
        case.set_u_prev(u_opt_ch)

        # call optimizer
        optimum = case.optimize()

        # get objective and design variables
        f_opt_ph = optimum['objective']
        u_opt_ph = optimum['variable']

        # get the control action for the control horizon
        u_opt_ch = u_opt_ph[0:case.n]

        # overwrite fan speed
        uFan = np.maximum(float(u_opt_ch[0]),0)

        # update start points for optimizer using previous optimum value
        case.set_u_start(u_opt_ph)

        # update predictions after MPC predictor is called otherwise use measurement 
        print(u_opt_ph, case._autoerror)
        Tz_pred = float(case.predict_zone_temp(
            case.states['Tz_his_meas'], case.states['To_his_meas'], u_opt_ch[0]*mFan_nominal, 14, case._autoerror))
        # update power prediction after MPC call
        P_pred = float(case.predict_power(u_opt_ch[0]*mFan_nominal, Tz_pred))

    ### advance building simulation by one step
    #u_traj = np.transpose(np.vstack(([ts,te],[uFan,uFan])))
    #input_object = ("uFan",u_traj)
    hvac.set("uFan",uFan)
    res = hvac.simulate(start_time = ts,
                        final_time = te, 
                        options = options)

    # update clock
    ts = te

    # get measurement
    measurement = get_measurement(res,measurement_names)
    print(measurement)

    # update MPC model states
    # if not warmup then measurement else from mpc
    if warmup:
        Tz_pred = measurement['TRoo'].values[0] - 273.15
        P_pred = measurement['PTot'].values[0]
        u_opt_ch = [uFan, 0.1]

    states = get_states(states,measurement, Tz_pred)
    print ("\nstate 4")
    print (states)

    # online MPC model calibration if applied - NOT IMPLEMENTED
    # update parameter_zones and parameters_power - NOT IMPLEMENTED
    
    # update predictor
    predictor['price'] = get_price(ts,dt,PH)
    predictor['Toa'] = get_Toa(ts,dt,PH,Toa_year)

    # update fmu settings
    options['initialize'] = False

    # update warmup flag for next step
    warmup = ts<te_warm

    # Save all the optimal results for future simulation
    u_opt.append(uFan)
    Tz_pred_opt.append(Tz_pred)
    P_pred_opt.append(P_pred)

final = {'u_opt':u_opt,
        't_opt':t_opt,
        'Tz_pred_opt':Tz_pred_opt,
        'P_pred_opt':P_pred_opt}

with open('u_opt.json', 'w') as outfile:
    json.dump(final, outfile)
