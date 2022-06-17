#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import global optimizer
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem
from pymoo.util.display import Display
# Import numerical libraries
import numpy as np
import numpy.matlib
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import json
import argparse
import os

# Import the needed JModelica.org Python methods
from pyfmi import load_fmu


class MyDisplay(Display):

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        fmin = algorithm.es.gi_frame.f_locals
        cma = fmin["es"]

        self.output.append("fopt", algorithm.opt[0].F[0])

        if fmin["restarts"] > 0:
            self.output.append("run", int(
                fmin["irun"] - fmin["runs_with_small"]) + 1, width=4)
            self.output.append("fpop", algorithm.pop.get("F").min())
            self.output.append("n_pop", int(cma.opts['popsize']), width=5)

        self.output.append("sigma", cma.sigma)

        val = cma.sigma_vec * cma.dC ** 0.5
        self.output.append("min std", (cma.sigma * min(val)), width=8)
        self.output.append("max std", (cma.sigma * max(val)), width=8)

        axis = (cma.D.max() / cma.D.min()
                if not cma.opts['CMA_diagonal'] or cma.countiter > cma.opts['CMA_diagonal']
                else max(cma.sigma_vec * 1) / min(cma.sigma_vec * 1))
        self.output.append("axis", axis, width=8)

        # temporarily save the best x found so far
        np.savetxt('x_opt.txt',algorithm.opt.get("X"))

class PerfectMPC(object):
    def __init__(self, 
                PH,
                CH,
                time,
                dt,
                fmu_model, 
                fmu_generator="jmodelica",
                measurement_names = [],
                control_names = [],
                price = [],
                u_lb = [],
                u_ub = []):
        # MPC settings
        self.PH = PH 
        self.CH = CH
        self.dt = dt 
        self.time = time

        # MPC models
        self.fmu_model = fmu_model
        self.fmu_generator = fmu_generator
        self.fmu_output_names = measurement_names
        self._fmu_results = [] # store intermediate fmu results during optimization
        self.fmu_options = self.fmu_model.simulate_options()
        self.fmu_options["result_handling"] = "memory"
        self.fmu_options["filter"] = self.fmu_output_names
        self.fmu_options["ncp"] = min(1000, int(self.dt/60.)*PH)

        # define fmu model inputs for optimization: here we assume we only optimize control inputs (time-varying variabels in Modelica instead of parameter)
        self.fmu_input_names = control_names
        self.ni = len(control_names)

        # some building settings
        self.occ_start = 8
        self.occ_end = 18

        # predictors
        self.price = price

        # control input bounds
        self.u_lb = u_lb
        self.u_ub = u_ub

        # mpc tuning
        self.weights = [100., 1.0, 10.0]
        self.u0 = [1.0]*self.PH
        self.u_ch_prev = self.u_lb

        # optimizer settings
        self.optimizer = "cmaes"
        self.resume_from_checkpoint = False
        self.checkpoint = "checkpoint.npy"

    def get_fmu_states(self):
        """ Return fmu states as a hash table"""

        return self._get_fmu_states()
    
    def _get_fmu_states(self):
        states = {}
        if self.fmu_generator == "jmodelica":
            names = self.fmu_model.get_states_list()
            for name in names:
                states[name] = float(self.fmu_model.get(name))
        elif self.fmu_generator == "dymola":
            states = self.fmu_model.get_fmu_state()
        else:
            ValueError("FMU Generator not supported")

        return states

    def set_mpc_states(self, states):
        self._states_ = states

    def set_fmu_states(self, states):
        """ Set fmu states"""
        self._set_fmu_states(states)

    def _set_fmu_states(self, states):
        if self.fmu_generator == "jmodelica":
            for name in states.keys():
                self.fmu_model.set(name, states[name])
        elif self.fmu_generator == "dymola":
            self.fmu_model.set_fmu_state(states)
        else:
            pass

    def set_time(self, time):
        self.set_fmu_time(time)
        self.set_mpc_time(time)

    def set_fmu_time(self, time):
        self.fmu_model.time = float(time)

    def set_mpc_time(self, time):
        self.time = time

    def initialize_fmu(self):
        self.fmu_model.setup_experiment(start_time = self.time)
        self.fmu_model.initialize()
        self.fmu_options['initialize'] = False

    def reset_fmu(self):
        """
        reset fmu and options

        """
        self.fmu_model.reset()
        self.fmu_model.time = self.time

    def simulate(self, start_time, final_time, input = None, states={}):
        """ simulate fmu from given states"""
        if states:
            self.set_fmu_states(states)

        # call simulate()    
        self._fmu_results = self.fmu_model.simulate(start_time, final_time, input=input, options=self.fmu_options)
        
        # return states and measurments
        states_next = self.get_fmu_states()
        outputs = self.get_fmu_outputs()

        return states_next, outputs

    def get_fmu_outputs(self):
        """ Get fmu outputs as dataframe for the objective function calculation"""
        outputs_dict = {}
        for output_name in self.fmu_output_names:
            outputs_dict[output_name] = self._fmu_results[output_name]

        time = self._fmu_results['time']
        outputs = pd.DataFrame(outputs_dict, index=time)

        return outputs

    def optimize(self):
        """ The optimization follows the following procedures:

            1. initialize fmu model and save initial states
            2. call optimizer to optimize the objective function and return optimal control inputs
            3. apply control inputs for next step, and return states and measurements
            4. recursively do step 2 and step 3 until the end time
        """
        u0 = self.u0 
        lower = self.u_lb*self.PH 
        upper =self.u_ub*self.PH 
        if self.optimizer == "cmaes":
            # Call instance of PSO
            optimizer = CMAES(
                x0 = np.array(u0),
                sigma = 0.25,
                normalize = True,
                parallelize = False,
                maxfevals = 50000,
                maxiter = 1000,
                tolfun = 1e-11,
                tolx = 1e-11,
                restarts = 0,
                restart_from_best=False,
                verb_log = 1,
            )

            obj = [self.objective_pymoo]
            c_ieq = []
            n_var = self.PH 

            prob = FunctionalProblem(
                n_var,
                obj,
                constr_ieq = c_ieq,
                xl = lower, 
                xu = upper)

            if self.resume_from_checkpoint:
                checkpoint, = np.load(self.checkpoint, allow_pickle=True).flatten()
                print("Loaded Checkpoint:", checkpoint)
                # only necessary if for the checkpoint the termination criterion has been met
                checkpoint.has_terminated = False
                out = minimize(prob,
                            checkpoint,
                            ('n_iter', 10000),
                            seed = 10,
                            copy_algorithm = False,
                            verbose = True)
                print(out.X, out.F)
            else:
            # Perform optimization
                out = minimize(
                    prob, 
                    optimizer, 
                    iters=5000, 
                    seed=10,
                    verbose=True,
                    display=MyDisplay(),
                    #save_history=True,
                    )
                print(out.X, out.F)
                # save as checkpoint in case
                np.save("checkpoint", optimizer)

        return out

    def objective_pymoo(self, u):
        """ return objective values at a prediction horizon for using pymoo
            
            1. transfer control variables generated from optimizer to fmu inputs
            2. call simulate() and return key measurements
            3. calculate and return MPC objective based on key measurements

        """
        # fetch simulation settings
        ts = self.time
        te = self.time + self.PH*self.dt

        # transfer inputs
        input_object = self._transfer_inputs(u, piecewise_constant=True)

        # call simulation
        self.reset_fmu()
        self.initialize_fmu()  # this might be a bottleneck for complex system
        _, outputs = self.simulate(
            ts, te, input=input_object, states=self._states_)

        # interpolate outputs as 1 min data and 15-min average
        t_intp = np.arange(ts, te+1, self.dt)
        outputs = self._sample(self._interpolate(outputs, t_intp), self.dt)
        # post processing the results to calculate objective terms
        energy_cost = self._calculate_energy_cost(outputs)
        temp_violations = self._calculate_temperature_violation(outputs)
        action_changes = self._calculate_action_changes(u)

        objective = [self.weights[0]*energy_cost[i] +
                     self.weights[1]*temp_violations[i]*temp_violations[i] +
                     self.weights[2]*action_changes[i]*action_changes[i] for i in range(self.PH)]

        print(sum(objective))

        return sum(objective)

    def _interpolate(self, df, new_index):
        """Interpolate a dataframe along its index based on a new index
        """
        df_out = pd.DataFrame(index=new_index)
        df_out.index.name = df.index.name

        for col_name, col in df.items():
            df_out[col_name] = np.interp(new_index, df.index, col)
        return df_out
    
    def _sample(self, df, freq):
        """ 
            assume df is interpolated at 1-min interval
        """
        index_sampled = np.arange(df.index[0], df.index[-1]+1, freq)
        df_sampled = df.groupby(df.index//freq).mean()
        df_sampled.index = index_sampled

        return df_sampled

    def _calculate_energy_cost(self, outputs):
        price = self.price # 24-hour prices
        t = outputs.index
        power = outputs['PTot']

        energy_cost = []
        for ti in t:
            h_index = int(ti % 86400/3600)
            energy_cost.append(power[ti]/1000*price[h_index]*self.dt/3600.)

        return energy_cost

    def _calculate_temperature_violation(self, outputs):
        # define nonlinear temperature constraints
        # zone temperature bounds - need check with the high-fidelty model
        T_upper = [30.0 for i in range(24)]
        T_upper[self.occ_start:self.occ_end] = [26.0]*(self.occ_end-self.occ_start)
        T_lower = [12.0 for i in range(24)]
        T_lower[self.occ_start:self.occ_end] = [22.0]*(self.occ_end-self.occ_start)

        # zone temperarture
        t = outputs.index
        temp = outputs['TRoo'] - 273.15

        overshoot = []
        undershoot= []
        violations = []
        for i, ti in enumerate(t):
            h_index = int(ti % 86400/3600)
            overshoot.append(max(temp[ti] - T_upper[h_index], 0.0))
            undershoot.append(max(T_lower[h_index] - temp[ti], 0.0))
            violations.append(overshoot[i]+undershoot[i])

        return violations

    def _calculate_action_changes(self, u):
        # # of inputs
        ni = self.ni
        # get bounds
        u_lb = self.u_lb
        u_ub = self.u_ub
        u_ch_prev = self.u_ch_prev

        du_nomalizer = [1./(u_ub[i] - u_lb[i]) for i in range(len(u_lb))]*self.PH
        du = []
        for i in range(self.PH):
            ui = u[i*ni:ni*(i+1)]
            dui_nomalizer = du_nomalizer[i*ni:ni*(i+1)]
            dui = [abs(ui[j] - u_ch_prev[j])*dui_nomalizer[j] for j in range(ni)] 
            u_ch_prev = ui
            du += dui

        return du

    def _transfer_inputs(self, u, piecewise_constant=False):
        """
        transfer optimizier outputs to fmu input objects

        :param u: optimization vectors from optimizer
        :type u: numpy.array or list
        """
        nu = len(u)
        ni = self.ni

        if nu != self.PH*ni:
            ValueError("The number of optimization variables are not equal to the number of control inputs !!!")
        
        ts = self.time 
        te = ts + self.PH*self.dt 
        u_trans = []
        t = np.arange(ts, te+1, self.dt).tolist()

        for i, input_name in enumerate(self.fmu_input_names):
            u_trans.append([u[(i+h*ni)]for h in range(self.PH)])
        t_input = t[:-1]

        # if need piecewise constant inputs instead of linear inpterpolation as default
        if piecewise_constant:
            t_piecewise = []
            # get piecewise constant time index
            for i, ti in enumerate(t[:-1]):
                t_piecewise += t[i:i+2]
            # get piecewise constant control input signals between steps
            u_trans_piecewise = []
            for j in u_trans:
                u_trans_j = []
                for i in range(len(j)):
                    u_trans_j += [j[i]]*2
                u_trans_piecewise.append(u_trans_j)

            t_input = t_piecewise
            u_trans = u_trans_piecewise
        # generate input object for fmu
        input_traj = np.transpose(np.vstack((t_input, u_trans)))
        input_object = (self.fmu_input_names, input_traj)
        
        return input_object

    def set_u0(self, u_ph_prev):
        """ set initial guess for the optimization at current step """
        self.u0 = list(u_ph_prev[self.ni:]) + self.u_lb

    def set_u_ch_prev(self, u_ch_prev):
        """save actions at previous step """
        self.u_ch_prev = u_ch_prev

def tune_mpc():
    fmu_generator = "jmodelica"
    fmu_path = os.path.dirname(os.path.realpath(__file__))
    print(fmu_path)
    if fmu_generator == "jmodelica":
        model = load_fmu(os.path.join(fmu_path, "SingleZoneFCU.fmu"))
    elif fmu_generator == "dymola":
        model = load_fmu(os.path.join(fmu_path, "SingleZoneFCUDymola.fmu"))

    PH = 672
    CH = 1
    dt = 900.
    ts = 201*24*3600.
    period = 7*24*3600.
    te = ts + period

    price = [0.02987, 0.02987, 0.02987, 0.02987,
             0.02987, 0.02987, 0.04667, 0.04667,
             0.04667, 0.04667, 0.04667, 0.04667,
             0.15877, 0.15877, 0.15877, 0.15877,
             0.15877, 0.15877, 0.15877, 0.04667,
             0.04667, 0.04667, 0.02987, 0.02987]

    measurement_names = ['uFan', 'PTot', 'TRoo']
    control_names = ['uFan']

    u_lb = [0.0]*len(control_names)
    u_ub = [1.0]*len(control_names)

    mpc = PerfectMPC(PH = PH,
                    CH = CH,
                    time = ts,
                    dt = dt,
                    fmu_model = model, 
                    fmu_generator = fmu_generator,
                    measurement_names = measurement_names,
                    control_names = control_names,
                    price = price,
                    u_lb = u_lb,
                    u_ub = u_ub)

    w_energy = 100.
    w_temp = 1.0
    w_action = 10.
    mpc.weights = [w_energy, w_temp, w_action]

    # update u0
    with open(os.path.join(fmu_path,'u0.json')) as f:
        u0 = json.load(f) 
    u0 = [i[0] for i in u0]
    mpc.u0 = u0[:PH]
    
    # resume optimiztion from checkpoint if needed
    mpc.resume_from_checkpoint = True
    mpc.checkpoint = "checkpoint.npy"

    # reset fmu
    mpc.reset_fmu()
    mpc.initialize_fmu()
    states = mpc.get_fmu_states()

    results = pd.DataFrame()
    u_opt = []
    t_opt = []

    t = ts

    # set starting states
    mpc.set_time(t)
    mpc.set_mpc_states(states)
    
    optimum = mpc.optimize()
    u_opt_ph = optimum.X 
    f_opt_ph = optimum.F
    u_opt_ch = u_opt_ph[:mpc.ni]
    
    # need revisit u0 design
    mpc.set_u0(u_opt_ph)
    mpc.set_u_ch_prev(u_opt_ch)

    # apply optimal control actions to virtual buildings
    mpc.reset_fmu()
    mpc.initialize_fmu()
    for i, name in enumerate(mpc.fmu_input_names):
        mpc.fmu_model.set(name, u_opt_ch[i])
    # simulate from saved states    
    states_next, outputs = mpc.simulate(t, t+dt, states=mpc._states_)

    # save results for future use
    u_opt.append([float(i) for i in u_opt_ph])
    results = pd.concat([results, outputs], axis=0)

    # update mpc clcok
    t += dt
    # update mpc states
    states = states_next

    # let's save the simualtion results 
    final = {'u_opt': u_opt}

    with open('u_opt.json', 'w') as outfile:
        json.dump(final, outfile) 

tune_mpc()
