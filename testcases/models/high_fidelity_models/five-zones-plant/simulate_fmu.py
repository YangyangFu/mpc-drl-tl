# -*- coding: utf-8 -*-
"""
this script is to test the simulation of compiled fmu
"""
from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

# import numerical package
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# import fmu package
from pyfmi import load_fmu
import numpy.random as random
import time 

def uniform(a,b):
    return (b-a)*random.random_sample()+a

# simulate setup
time_stop = 7*24*3600. 
startTime = 204*24*3600.
endTime = startTime + time_stop
dt = 60*15.

# define some filters to save simulation time using fmu
measurement_names = ['time','TZoneAirDev_y','TOutAir_y','GHI_y','PHVAC_y','yFanSpe_y','yDamMax_y',
'yDamMin_y','yWatVal_y','yCooTowFan_y','modCoo.conAHU.TSup','modCoo.conAHU.TSupSet','modCoo.eleTot.y','oveAct_TSupSet']

## load fmu - cs
fmu_name = "FiveZoneSystemSim"
fmu = load_fmu(fmu_name+'.fmu')
fmu.set_log_level(0) # log level 0-7
options = fmu.simulate_options()
options['filter']=measurement_names
options['result_handling']="memory" #"memory"

options['ncp'] = 100

# initialize output
y = []
tim = []

# input: None
# Get input names
input_names = fmu.get_model_variables(causality = 2).keys()
print(input_names)
print('Inputs: {0}'.format(input_names))

# simulate fmu
initialize = True
res_all=[]

ts = startTime
tic = time.process_time()
while ts < endTime:
    # settings
    te = ts + dt
    options['initialize'] = initialize  
    # generate inputs
    u = uniform(12+273.15,18+273.15)
    v = uniform(5+273.15,10+273.15)
    w = uniform(18000,36000)
    data = np.transpose(np.vstack(([ts,ts+dt],[u,u],[v,v],[w,w])))
    print ("data", data)
    input_object = (list(input_names),data)
    res_step = fmu.simulate(start_time=ts, final_time=te, options=options, input = input_object)
    print ("simulated")
    res_all.append(res_step)
    initialize = False
    ts = te

    # get results

toc = time.process_time()

print ('Finish simulation in:' + str(toc-tic)+" second(s)")

measurement_base={}

for name in measurement_names:
    value_name=[]
    for res in res_all:
        value_name += list(res[name])
    measurement_base[name] = np.array(value_name)

   
plt.figure(figsize=(8,10))
plt.subplot(211)
plt.plot(measurement_base['time'],measurement_base['oveAct_TSupSet'],'b-',label="TSup")
plt.plot(measurement_base['time'],measurement_base['modCoo.conAHU.TSupSet'],'r--',label="TSupSet")
plt.legend()
plt.ylabel('TSupSet')
plt.subplot(212)
plt.plot(measurement_base['time'],measurement_base['modCoo.eleTot.y'])
plt.ylabel('PHVAC')
plt.savefig('simulateFMUInputs.pdf')


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
        
filelist = [fmu_name+'_result.mat',fmu_name+'_log.txt']
deleteFiles(filelist)