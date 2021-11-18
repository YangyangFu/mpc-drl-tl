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

# simulation setup
ts = 222*24*3600.#+13*24*3600
nday = 7
period = nday*24*3600.
te = ts + period
dt = 15*60.
nsteps_h = int(3600//dt)

# setup for DRL test
nActions = 51
alpha = 1
nepochs = 500

# define some filters to save simulation time using fmu
measurement_names = ['time','PHVAC','PBoiGas','TRooAirSou','TRooAirEas','TRooAirNor','TRooAirWes','TRooAirCor','TRooAirDevTot','EHVACTot','conAHU.TSupSet','conAHU.TSup','uTSupSet']


baseline = load_fmu('FiveZoneVAVBaseline.fmu')

print ("test1")

## fmu settings
options = baseline.simulate_options()
options['ncp'] = 5000
options['initialize'] = True
options['result_handling'] = 'memory'
options['filter'] = measurement_names

print ("Done!")

## construct optimal input for fmu
res_base = baseline.simulate(start_time = ts,
                    final_time = te, 
                    options = options)

print ("Finish baseline simulation")



