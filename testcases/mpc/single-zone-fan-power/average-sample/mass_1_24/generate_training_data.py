# -*- coding: utf-8 -*-
"""
this script is to test the simulation of compiled fmu
"""
from __future__ import print_function
from __future__ import absolute_import, division
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
time_stop = 31*24*3600.  
ts = 181*24*3600.
te = ts + time_stop

## load fmu - cs
fmu_name = "SingleZoneFCU.fmu"
fmu = load_fmu(fmu_name)
options = fmu.simulate_options()
options['ncp'] = 10000
fmu.set_log_level(7)

# excite signal: - generator for exciting signals
def uniform(a,b):
    return (b-a)*random.random_sample()+a

def excite_fan(time):
    y = np.zeros(time.shape)
    j = 0
    for i in time:
        h = int((i%86400)/3600)
        if h<6:
             y[j] = 0.
        elif h<19:
            y[j] = uniform(0,1) 
        else:
            y[j] = 0.         
        j+=1
    return y

# generate signal for every dt
dt = 15*60. 
time_arr = np.arange(ts,te+1,dt)
spe_sig = excite_fan(time_arr)


# input
input_trac = np.transpose(np.vstack((time_arr,spe_sig)))
input_object = ('uFan',input_trac)

# simulate fmu
res = fmu.simulate(start_time=ts,
                    final_time=te, 
                    input=input_object,
                    options=options)

# what data do we need??
tim = res['time']
spe = res['uFan']
flo = res['m_flow_in']
Toa = res['TOut']
Tsa = res['fcu.TSup']
TRoo = res['TRoo']
PTot = res['PTot']

# interpolate data
train_data = pd.DataFrame({'speed':np.array(spe),
                            'mass_flow':np.array(flo),
                            'T_oa':np.array(Toa),
                            'T_sa':np.array(Tsa),
                            'T_roo':np.array(TRoo),
                            'P_tot':np.array(PTot)}, index=tim)

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
