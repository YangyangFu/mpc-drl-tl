# -*- coding: utf-8 -*-
"""
this script is to test the simulation of compiled fmu
"""
# import numerical package
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# import fmu package
from pyfmi import load_fmu
from pymodelica import compile_fmu


# simulate setup
time_stop = 120. # 120s
startTime = 0
endTime = startTime + time_stop
dt = 60

## compile fmu - cs
mopath = 'SingleZoneVAV.mo'
modelpath = 'SingleZoneVAV.Airflow'
fmu_name = "SingleZoneVAV"
fmu = load_fmu(compile_fmu(modelpath,[mopath], target='cs',version='2.0',compile_to=fmu_name+'.fmu'))
options = fmu.simulate_options()
options['ncp'] = 500.

# initialize output
y = []
tim = []

# input: None
# Get input names
input_names = fmu.get_model_variables(causality = 2).keys()
print input_names
print('Inputs: {0}'.format(input_names))

timeArray = np.arange(startTime,endTime,dt)
uFan = 0.5*np.ones(timeArray.shape)
u=np.transpose([timeArray.flatten(),uFan.flatten()])
# input object as defined for jmdoelica fmu
input_object = (input_names, u)

# simulate fmu
initialize = True

for i in range(2):
    ts = i*dt
    options['initialize'] = initialize    

    res_step = fmu.simulate(input = input_object,start_time=ts, final_time=ts+dt, options=options)
    initialize = False
    print res_step['TRoo']-273.15

print 'Finish simulation'

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