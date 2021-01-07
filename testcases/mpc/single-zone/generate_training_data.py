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
# import fmu package


# simulate setup
time_stop = 30*24*3600.  # 120s
startTime = 212*24*3600.
endTime = startTime + time_stop

## load fmu - cs
fmu_name = "SingleZoneVAVBaseline.fmu"
fmu = load_fmu(fmu_name)
options = fmu.simulate_options()
options['ncp'] = 500.

# initialize output
y = []
tim = []

# input: None

# simulate fmu
res = fmu.simulate(start_time=startTime,
                    final_time=endTime, options=options)

# what data do we need??


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
