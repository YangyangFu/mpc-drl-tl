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
time_stop = 24*3600. 
startTime = 0
endTime = startTime + time_stop
dt = 60*15.

## load fmu - cs
fmu_name = "SingleZoneDamperControl"
fmu = load_fmu(fmu_name+'.fmu')
fmu.set_log_level(0) # log level 0-7
options = fmu.simulate_options()
options['filter']=['uFan','TRoo','hvac.uFan']
options['result_handling']="memory" #"memory"

options['ncp'] = 100.

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

ts = startTime
tic = time.clock()
while ts < endTime:
    # settings
    te = ts + dt
    options['initialize'] = initialize  

    # generate inputs
    u = uniform(0.1,1)
    #time = ts
    #u_trajectory = np.vstack((time,u))
    #input_object = (input_names,np.transpose(u_trajectory))
    fmu.set(list(input_names),list([u]))
    res_step = fmu.simulate(start_time=ts, final_time=te, options=options)

    initialize = False
    ts = te

    # get results

toc = time.clock()

print ('Finish simulation in:' + str(toc-tic)+" second(s)")

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