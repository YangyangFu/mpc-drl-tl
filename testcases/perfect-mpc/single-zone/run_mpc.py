#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import library for path manipulations
import os.path

# Import numerical libraries
import numpy as N
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Import the needed JModelica.org Python methods
import pymodelica
from pymodelica import compile_fmu
from pyfmi import load_fmu
from pyjmi import transfer_optimization_problem
from pyjmi.optimization.mpc import MPC
from pyjmi.optimization.casadi_collocation import BlockingFactors
pymodelica.environ['JVM_ARGS'] = '-Xmx4g'

### 1. Compute initial guess trajectories by means of simulation
# Locate the Modelica and Optimica code
file_path = "./SingleZoneFanCoilUnit.mop"

# Compile and load the model used for simulation
#sim_fmu = compile_fmu("SingleZoneFanCoilUnit.TestCases.FanControlMPCModel", file_path,
#                        compiler_options={"state_initial_equations": True})
sim_fmu = "SingleZoneFanCoilUnit_TestCases_FanControlMPCModel.fmu"
sim_model = load_fmu(sim_fmu)

# Define stationary point A and set initial values and inputs
# TODO: check if we can use get_fmu_state or set_fmu_state here instead of manually extrac all state points
sim_model.get("zon.roo.air.vol.dynBal.U")
U_0_A = sim_model.get("zon.roo.air.vol.dynBal.U")[0]
fan_0_A = sim_model.get("fcu.fan.filter.x[2]")[0]

states = sim_model.get_states_list()
state_names = [state for state in states]
print(state_names)

sim_model.set('_start_zon.roo.air.vol.dynBal.U', U_0_A)
sim_model.set('_start_fcu.fan.filter.x[2]', fan_0_A)
sim_model.set('uFan', 0.2)
init_res = sim_model.simulate(start_time=203*24*3600., final_time=204*24*3600.)

### 2. Define the optimal control problem and solve it using the MPC class
# Compile and load optimization problem
op = transfer_optimization_problem("SingleZoneFanCoilUnit.TestCases.FanControlMPC", file_path,  compiler_log_level="debug", compiler_options={"state_initial_equations": True}, accept_model=True)

# Define MPC options
startTime= 203*24*3600
sample_period = 900.                           # s
horizon = 4*900                               # Samples on the horizon
n_e_per_sample = 1                          # Collocation elements / sample
n_e = n_e_per_sample*horizon                # Total collocation elements
finalTime = 204*24*3600                             # s
# Total number of samples to do
number_samp_tot = int((finalTime-startTime)/sample_period)

# Create blocking factors with quadratic penalty and bound on 'Tc'
bf_list = [n_e_per_sample]*(horizon/n_e_per_sample)
factors = {'Tc': bf_list}
du_quad_pen = {'Tc': 500}
du_bounds = {'Tc': 30}
bf = BlockingFactors(factors, du_bounds, du_quad_pen)

# Set collocation options
opt_opts = op.optimize_options()
opt_opts['n_e'] = n_e
opt_opts['n_cp'] = 2
opt_opts['init_traj'] = init_res
opt_opts['blocking_factors'] = bf

constr_viol_costs = {'T': 1e6}
# Create the MPC object
MPC_object = MPC(op, opt_opts, sample_period, horizon,
                    constr_viol_costs=constr_viol_costs, noise_seed=1)

# Set initial state
x_k = {'_start_zon.roo.air.vol.dynBal.U': U_0_A, '_start_fcu.fan.filter.x[2]': fan_0_A}
#print(dir(MPC_object))
# Update the state and optimize number_samp_tot times
for k in range(number_samp_tot):
    # Update the state and compute the optimal input for next sample period
    MPC_object.update_state(x_k)
    u_k = MPC_object.sample()
    
    # Reset the model and set the new initial states before simulating
    # the next sample period with the optimal input u_k
    sim_model.reset()
    sim_model.set(x_k.keys(), x_k.values())
    sim_res = sim_model.simulate(start_time=k*sample_period,
                                    final_time=(k+1)*sample_period,
                                    input=u_k)

    # Extract state at end of sample_period from sim_res and add Gaussian
    # noise with mean 0 and standard deviation 0.005*(state_current_value)
    x_k = MPC_object.extract_states(sim_res, mean=0, st_dev=0.005)

# Extract variable profiles
MPC_object.print_solver_stats()
