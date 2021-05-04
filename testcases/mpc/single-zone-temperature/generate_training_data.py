# -*- coding: utf-8 -*-
"""
this script is to test the simulation of compiled fmu
"""
# import numerical package
#from pymodelica import compile_fmu
from pyfmi import load_fmu
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import pandas as pd
# import fmu package


# simulate setup - 181-212 for july; 212-243 for August
time_stop = 30*24*3600.  
ts = 212*24*3600.
te = ts + time_stop

## load fmu - cs
fmu_name = "SingleZoneTemperature.fmu"
fmu = load_fmu(fmu_name)
options = fmu.simulate_options()
options['ncp'] = 10000.

# excite signal: - generator for exciting signals
def uniform_real(a,b):
    return (b-a)*random.random_sample()+a

def uniform_int(a,b):
    return random.randint(a,b)

def excite(time, signals, min_dur, max_dur):
    """
    psuedo code:
    excite_flag = true
    for time t:
        if excite_flag:
            randomly generate a set of values for each signal
            randomly generate a set of durations for each signal
            use the generated signal and durations
            update excite flag for each signal
                compare excitation end time and next step
            update previous signal
        else:
            use previous signal
            update excite flag
                ends of current excited period
    """
    # for each signal
    N = len(time)
    # initialize flag for each signal
    for key in signals.keys():
        signals[key]['flag'] = True

    # initialize a data frame for storing excited signals
    col_names = signals.keys()
    excited_signals = pd.DataFrame(columns=col_names, index=time)
    # main loop
    for i in range(N):
        t = time[i]
        t_next = time[min(i+1,N-1)] 
        h = int((t%86400)/3600)            
        # generate signal/duration for each signal
        for key in signals.keys():
            min_sig = signals[key]['min']
            max_sig = signals[key]['max']
            typ_sig = signals[key]['type']
            flag = signals[key]['flag']
            if flag:
                # generate excite signal
                if typ_sig == 'real':
                    if h<7 or h>=19:
                        sig_t = uniform_real(min_sig, max_sig)
                        sig_dur = uniform_int(min_dur, max_dur+1)
                    else:
                        sig_t = uniform_real(273.15+20, max_sig)
                        sig_dur = uniform_int(min_dur, max_dur+1)
                else:
                    sig_t = uniform_int(min_sig, max_sig+1)
                    sig_dur = uniform_int(min_dur, max_dur+1)
                print key+': '+str(sig_dur)
                # check excite flag and update excited signal
                excited_signals.loc[t,key] = sig_t

                # update excite flag
                t_exc_end = time[min(i+sig_dur,N-1)]

                if t_next< t_exc_end:
                    signals[key]['flag']= False

                # update previous signal and endtime of current excitation duration
                signals[key]['signal_prev'] = sig_t
                signals[key]['excite_endtime'] = t_exc_end
            else: # dont excite instead we use previous excited signal till the end of excited duration
         
                excited_signals.loc[t,key] = signals[key]['signal_prev'] 
                # update flag
                t_exc_end = signals[key]['excite_endtime']
                if t_next>= t_exc_end:
                    signals[key]['flag']= True

    return excited_signals

# generate signal for every dt
dt = 15*60. 
time_arr = np.arange(ts,te+1,dt)
control_sig = {"TSetCoo":{'min':273.15+12, 'max':273.15+34,'type':'real'}}
sig_df = excite(time_arr,control_sig,4,13)
sig_values = sig_df["TSetCoo"].values

# input
input_names = fmu.get_model_variables(causality=2).keys()
input_trac = np.transpose(np.vstack((time_arr.flatten(),sig_values.flatten()))).astype('float64')
input_object = (input_names,input_trac)

# simulate fmu
res = fmu.simulate(start_time=ts,
                    final_time=te, 
                    options=options,
                    input = input_object)

# what data do we need??
tim = res['time']
TCooSet = res['TSetCoo']
Toa = res['TOut']
TRoo = res['TRoo']
PTot = res['PTot']
PCoo = res['PCoo.y']
PHea = res['PHea.y']
PPum = res['PPum.y']
PFan = res['PFan.y']

# interpolate data
train_data = pd.DataFrame({'T_set':np.array(TCooSet),
                            'T_oa':np.array(Toa),
                            'T_roo':np.array(TRoo),
                            'P_tot':np.array(PTot),
                            'P_coo':np.array(PCoo),
                            'P_pum':np.array(PPum),
                            'P_fan':np.array(PFan),
                            'P_hea':np.array(PHea)}, index=tim)

def interp(df, new_index):
    """Return a new DataFrame with all columns values interpolated
    to the new_index values."""
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name

    for colname, col in df.iteritems():
        df_out[colname] = np.interp(new_index, df.index, col)

    return df_out

#interpolate one minute data
train_data_tim = np.arange(ts,te+1,60) 
train_data = interp(train_data, train_data_tim)
#average every 15 minutes
train_data_15 = train_data.groupby(train_data.index//900).mean()
train_data_15.to_csv('train_data.csv')

# clean folder after simulation
def deleteFiles(fileList):
    """ Deletes the output files of the simulator.

    :param fileList: List of files to be deleted.

    """
    import os

    for fil in fileList:
        try:
            if os.path.exists(fil):
                os.remove(fil)
        except OSError as e:
            print ("Failed to delete '" + fil + "' : " + e.strerror)


filelist = [fmu_name+'_result.mat', fmu_name+'_log.txt']
deleteFiles(filelist)
