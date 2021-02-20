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
    return pd.DataFrame(dic,index=dic['time'])
    
def interpolate_dataframe(df,new_index):
    """Interpolate a dataframe along its index based on a new index
    """
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name

    for col_name, col in df.items():
        df_out[col_name] = np.interp(new_index, df.index, col)    
    return df_out

def FILO(array_1d,x):
    """First in last out
    """
    array_list=list(array_1d)
    array_list.pop() # remove the last element
    array_list.reverse()
    array_list.append(x)
    array_list.reverse()
    array=np.array(array_list)

    return array

def get_states(states,measurement):
    Tz_his = states['Tz_his_t']
    P_his = states['P_his_t']

    Tz = measurement['TRoo']
    P = measurement['PTot']

    new_Tz_his = FILO(Tz_his,Tz)
    new_P_his = FILO(P_his,P)
    
    states['Tz_his_t'] = new_Tz_his
    states['P_his_t']= new_P_his

    return states

def get_price(time,dt,PH):
    price_tou = [0.0640, 0.0640, 0.0640, 0.0640, 
        0.0640, 0.0640, 0.0640, 0.0640, 
        0.1391, 0.1391, 0.1391, 0.1391, 
        0.3548, 0.3548, 0.3548, 0.3548, 
        0.3548, 0.3548, 0.1391, 0.1391, 
        0.1391, 0.1391, 0.1391, 0.0640]
    #- assume hourly TOU pricing
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

    tem_sol_h = dat[0][['temp_air']]+273.15
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
start = 212*24*3600.
end = start + 1*24*3600.

### 1- Load virtual building model
hvac = load_fmu('SingleZoneVAV.fmu')

## fmu settings
options = hvac.simulate_options()
options['ncp'] = 500.

# Warm up FMU simulation settings
ts = start
te_warm = ts + 4*3600

### 2- Initialize MPC case 
dt = 15*60.
PH = 4*24
CH = 1
with open('TZone.json') as f:
  parameters_zone = json.load(f)

with open('PTotal.json') as f:
  parameters_power = json.load(f)

# measurement at current time
measurement_names=['TRoo','TOut','PTot','uFan','hvac.fanSup.m_flow_in']
measurement_ini = {}
# states at current time for MPC model - this should be customized based on mpc design
lag_Tz = 4 # 4-step lag - should be identified for MPC model
lag_PTot = 4 # 4-step lag - should be idetified for mpc model
Tz_ini = 273.15+20
P_ini = 0.0
states_ini = {'Tz_his_t':np.array([Tz_ini]*lag_Tz),
            'P_his_t':np.array([P_ini]*lag_PTot)} # initial states used for MPC models

## predictors
predictor = {}

# energy prices 
predictor['price'] = get_price(ts,dt,PH)
# outdoor air temperature
weather_file = 'USA_CA_Riverside.Muni.AP.722869_TMY3.epw'
Toa_year = read_temperature(weather_file,dt)
predictor['Toa'] = get_Toa(ts,dt,PH,Toa_year)

### 3- MPC Control Loop
uFan_ini = 0.1
# initialize fan speed for warmup setup
uFan = uFan_ini
states = states_ini
measurement = measurement_ini
# initialize mpc case 
case = mpc_case(PH=PH,
                CH=CH,
                time=ts,
                dt=dt,
                parameters_zone = parameters_zone,
                parameters_power = parameters_power,
                measurement = measurement,
                states = states,
                predictor = predictor)

while ts<end:
    
    ### generate control action from MPC
    if ts>=te_warm: # activate mpc after warmup
        # update mpc case
        case.set_time(ts)
        case.set_measurement(measurement)
        case.set_states(states)
        case.set_predictor(predictor)
        # call optimizer
        optimum = case.optimize()

        # get objective and design variables
        f_opt_ph = optimum['objective']
        u_opt_ph = optimum['variable']
        
        # get the control action for the control horizon
        u_opt_ch = u_opt_ph[0]

        # overwrite fan speed
        uFan = u_opt_ch

    ### advance building simulation by one step
    u_traj = np.transpose(np.vstack(([ts,ts+dt],[uFan,uFan])))
    input_object = ("uFan",u_traj)
    res = hvac.simulate(start_time = ts,
                        final_time = ts+dt, 
                        options = options,
                        input = input_object)
    # get measurement
    measurement = get_measurement(res,measurement_names)

    # update MPC model inputs
    states = get_states(states,measurement)

    # online MPC model calibration if applied - NOT IMPLEMENTED
    # update parameter_zones and parameters_power - NOT IMPLEMENTED
    
    # update clock
    ts += dt

    # update predictor
    predictor['price'] = get_price(ts,dt,PH)
    predictor['Toa'] = get_Toa(ts,dt,PH,Toa_year)
