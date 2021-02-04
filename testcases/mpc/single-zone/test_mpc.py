import numpy as np
import pandas as pd

# import mpc models
from models import total_power
from models import zone_temperature
# load testbed
from pyfmi import load_fmu

###: need add a price generator
price = []

### Simulation setup
start = 212*24*3600.
end = start + 1*24*3600
dt = 15*60.


### Load virtual building model
hvac = load_fmu('SingleZoneVAV.fmu')

## fmu settings
options = hvac.simulate_options()
options['ncp'] = 500.

### 
#
ts = start
te_warm = ts + 4*3600

# Warm up FMU simulation
res_wram=hvac.simulate(start_time=ts,
            final_time=te_warm, 
            options=options,
            input = {'uFan':0.1})

ts = te_warm

# MPC Control Loop
while ts<end:
    
    # hour index
    hour = int((ts%86400)/3600)

    # get measurement

    # online MPC model calibration if applied - NOT IMPLEMENTED

    # perform MPC calculation
    #  - call mpc optimizer

    # Implement action to control horizon


        
    # update clock
    ts += dt
