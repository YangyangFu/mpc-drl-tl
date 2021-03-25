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
from pyfmi.common.io import ResultDymolaBinary

res = ResultDymolaBinary("training_data.mat")

train_data = pd.DataFrame()
meas_names = ['uFan','hvac.fanSup.m_flow_in','TOut','TRoo','PTot']
df_names = ['speed','mass_flow','T_oa','T_roo','P_tot']

for meas,col in zip(meas_names,df_names):
    var = res.get_variable_data(meas)
    tim=np.array(var.t)
    train_data[col]=np.array(var.x)

print tim
print train_data
train_data.index=tim
print train_data
# simulate setup - 181-212 for july; 212-243 for August
time_stop = 30*24*3600.  
ts = 212*24*3600.
te = ts + time_stop
print ts
print te
# generate signal for every dt
dt = 15*60. 

def interp(df, new_index):
    """Return a new DataFrame with all columns values interpolated
    to the new_index values."""
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name

    for colname, col in df.iteritems():
        df_out[colname] = np.interp(new_index, df.index, col)

    return df_out

#interpolate 15 minute data
train_data_tim = np.arange(ts,te+1,60*15) 
train_data = interp(train_data, train_data_tim)
train_data.to_csv('train_data_instant.csv')

