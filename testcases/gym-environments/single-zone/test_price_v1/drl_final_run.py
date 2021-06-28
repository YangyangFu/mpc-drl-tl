from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import json
# load testbed
from pyfmi import load_fmu

#==========================================================
##              General Settings
# =======================================================
# simulation setup
ts = 212*24*3600.#+13*24*3600
nday = 7
period = nday*24*3600.
te = ts + period
dt = 15*60.
nsteps_h = int(3600//dt)

# setup for DRL test
nActions = 11
alpha = 200
nepochs = 20

# define some filters to save simulation time using fmu

###############################################################
##              DRL final run
##===========================================================
# get actions from the last epoch
actions= np.load('./dqn/history_Action.npy')
u_opt = np.array(actions[-1,:,0])/float(nActions)
print (u_opt)

## fmu settings
hvac = load_fmu('SingleZoneDamperControl1.fmu')
options = hvac.simulate_options()
options['ncp'] = 5000.
options['initialize'] = True
#options['result_handling'] = 'memory'
#options['filter'] = measurement_names
res_drl = []

## construct optimal input for fmu
# 
u = u_opt
t = np.arange(ts, te, dt)
u_traj = np.transpose(np.vstack((t,u)))
input_object = ("uFan",u_traj)
ires = hvac.simulate(start_time = ts,
            final_time = te, 
            options = options,
            input = input_object)


