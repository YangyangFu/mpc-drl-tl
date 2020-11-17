# -*- coding: utf-8 -*-
"""
This module compiles the defined test case model into an FMU using the
overwrite block parser.

The following libraries must be on the MODELICAPATH:

- Modelica IBPSA
- Modelica Buildings

"""
# import numerical package
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# import fmu package
from pyfmi import load_fmu
# import buildingspy


## load fmu
name = "DataCenterExportFMUCS.fmu"
fmu = load_fmu(name)
# Get input names
input_names = fmu.get_model_variables(causality = 2).keys()
print input_names
print('Inputs: {0}'.format(input_names))


# simulate setup
startTime = 182*24*3600.
endTime = 183*24*3600.
dt = 3600.

# input
timeArray = np.arange(startTime,endTime,dt)
uc_set = 0.8*2000000*np.ones(timeArray.shape)
us_set = 0.*np.ones(timeArray.shape)

u=np.transpose([timeArray.flatten(),us_set.flatten(),uc_set.flatten()])

# simulate
opts = fmu.simulate_options()
initialize = True
ts = startTime

while ts < endTime:

    finalTime = ts + dt
    res = fmu.simulate(input=(input_names, u),start_time = ts, final_time=finalTime, options=opts)

    # update clock
    ts = finalTime
    # update initialization
    initialize = False
    opts['initialize'] = initialize

    #res = fmu.simulate(start_time = startTime, final_time=endTime, options=opts)
    # read output
    t = res['time']
    uc = res['uc_set']
    eneCost = res['eneCosAll.y']
    TAirRoo = res['dc.roo.TRooAir']
    print(t)
    print(uc)
    print(eneCost)
    print(TAirRoo)
