from __future__ import print_function
from __future__ import absolute_import, division

import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import json
# load testbed
from pyfmi import load_fmu
import time 

#==========================================================
##              General Settings
# =======================================================
# simulate setup
time_stop = 7*24*3600. 
startTime = 159*24*3600.
endTime = startTime + time_stop
dt = 60*15.

# define some filters to save simulation time using fmu
measurement_names = ['time','eleTot.y','conAHU.TSup','conAHU.TSupSet','conAHU.ducStaPre','conAHU.supFan.conSpe.u_s']

## load fmu - cs
fmu_name = "FiveZoneAirBaselineSho"
fmu = load_fmu(fmu_name+'.fmu')
fmu.set_log_level(0) # log level 0-7
options = fmu.simulate_options()
options['filter']=measurement_names
options['result_handling']="memory" #"memory"

options['ncp'] = 100

# initialize output
y = []
tim = []

# simulate fmu
initialize = True
res_all=[]

ts = startTime
tic = time.process_time()
while ts < endTime:
    # settings
    te = ts + dt
    options['initialize'] = initialize  
    res_step = fmu.simulate(start_time=ts, final_time=te, options=options)
    res_all.append(res_step)
    initialize = False
    ts = te

toc = time.process_time()

print ('Finish simulation in:' + str(toc-tic)+" second(s)")

measurement_base={}

for name in measurement_names:
    value_name=[]
    for res in res_all:
        value_name += list(res[name])
    measurement_base[name] = np.array(value_name)

from buildingspy.io.outputfile import Reader

r1 = Reader("SystemShoulderSeasonBaseline.mat", "dymola")

(time1, TSup) = r1.values("conAHU.TSup")
(time1, ducStaPre) = r1.values("conAHU.ducStaPre")
(time1, eleTot) = r1.values("eleTot.y")

plt.figure(figsize=(8,12))
plt.subplot(311)
plt.plot(measurement_base['time'],measurement_base['conAHU.TSup'],'b-',label="TSup")
plt.plot(time1,TSup,'r-',label="TSup Dymola")
#plt.plot(measurement_base['time'],measurement_base['conAHU.TSupSet'],'r--',label="TSupSet")
plt.legend()
plt.ylabel('TSupSet')
plt.subplot(312)
plt.plot(measurement_base['time'],measurement_base['conAHU.ducStaPre'],'b-',label="dp")
plt.plot(time1,ducStaPre,'r-',label="dp Dymola")
#plt.plot(measurement_base['time'],measurement_base['conAHU.supFan.conSpe.u_s']*410,'r--',label="TSupSet")
plt.legend()
plt.ylabel('dpSet')
plt.subplot(313)
plt.plot(measurement_base['time'],measurement_base['eleTot.y'],'b-')
plt.plot(time1,eleTot,'r-')
plt.ylabel('PHVAC')
plt.savefig('simulateBaselineCompare.pdf')


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