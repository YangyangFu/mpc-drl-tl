#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pybobyqa

# Import numerical libraries
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Import the needed JModelica.org Python methods
from pyfmi import load_fmu

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
                price = []):
        # MPC settings
        self.PH = PH 
        self.CH = CH
        self.dt = dt 
        self.time = time

        # MPC models
        self.fmu_model = fmu_model
        self.fmu_generator = fmu_generator

        self.fmu_options = self.fmu_model.simulate_options()
        self.fmu_options["result_handling"] = "memory"

        self.fmu_output_names = measurement_names
        self._fmu_results = [] # store intermediate fmu results during optimization
        
        # define fmu model inputs for optimization: here we assume we only optimize control inputs (time-varying variabels in Modelica instead of parameter)
        self.fmu_input_names = control_names

        # some building settings
        self.occ_start = 8
        self.occ_end = 18

        # predictors
        self.price = price

        # control input bounds
        self.u_lb = [0.0]
        self.u_ub = [1.0]

        # mpc tuning
        self.weights = [100., 1.0, 0.]
        self.u0 = [1.0]*self.PH
        self.u_prev = [0.0]

    def get_fmu_states(self):
        """ Return fmu states as a hash table"""

        return self._get_fmu_states()
    
    def _get_fmu_states(self):
        states = {}
        if self.fmu_generator == "jmodelica":
            names = self.fmu_model.get_states_list()
            for name in names:
                states[name] = float(self.fmu_model.get(name))

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
            pass 

    def set_time(self, time):
        self.fmu_model.time = float(time)

    def initialize_fmu(self):
        self.fmu_model.initialize()
        self.fmu_options['initialize'] = False

    def reset_fmu(self):
        """
        reset fmu and options

        """
        self.fmu_model.reset()
        self.fmu_options = self.fmu_model.simulate_options()
        self.fmu_options["result_handling"] = "memory"

    def simulate(self, start_time, final_time, input = None, states={}):
        """ simulate fmu from given states"""
        if states:
            self.set_fmu_states(states)

        # filter: list of output variabels for objective calculation
        self.fmu_options['filter'] = self.fmu_output_names

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

        soln = pybobyqa.solve(self.objective, u0, maxfun=1000, bounds=(
            lower, upper), seek_global_minimum=False)
        print(soln)

    def objective(self, u):
        """ return objective values at a prediction horizon
            
            1. transfer control variables generated from optimizer to fmu inputs
            2. call simulate() and return key measurements
            3. calculate and return MPC objective based on key measurements

        """
        # fetch simulation settings
        ts = self.time 
        te = self.time + self.PH*self.dt
        
        # transfer inputs
        input_object = self._transfer_inputs(u, piecewise_constant=True)
        _, outputs = self.simulate(ts, te, input=input_object, states=self._states_)
        self.reset_fmu()

        # interpolate outputs as 1 min data and 15-min average
        t_intp = np.arange(ts, te, 60)
        outputs = self._sample(self._interpolate(outputs, t_intp), self.dt)
        # post processing the results to calculate objective terms
        energy_cost = self._calculate_energy_cost(outputs)
        temp_violations = self._calculate_temperature_violation(outputs)
        action_changes = self._calculate_action_changes(u)

        objective = [self.weights[0]*energy_cost[i] + \
                     self.weights[1]*temp_violations[i]*temp_violations[i] +
                     self.weights[2]*action_changes[i]*action_changes[i] for i in range(self.PH)]
        
        print("")
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
        index_sampled = np.arange(df.index[0], df.index[-1], freq)
        df_sampled = df.groupby(df.index//freq).mean()
        df_sampled.index = index_sampled

        return df_sampled

    def _calculate_energy_cost(self, outputs):
        price = self.price # 24-hour prices
        t = outputs.index
        power = outputs['PTot']

        energy_cost = []
        print(t)
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
        ni = len(self.fmu_input_names)
        # get bounds
        u_lb = self.u_lb
        u_ub = self.u_ub
        u_prev = self.u_prev

        du_nomalizer = [1./(u_ub[i] - u_lb[i]) for i in range(len(u_lb))]*self.PH
        du = []
        for i in range(self.PH):
            ui = u[i*ni:ni*(i+1)]
            dui_nomalizer = du_nomalizer[i*ni:ni*(i+1)]
            dui = [abs(ui[j] - u_prev[j])*dui_nomalizer[j] for j in range(ni)] 
            u_prev = ui
            du += dui

        return du

    def _transfer_inputs(self, u, piecewise_constant=False):
        """
        transfer optimizier outputs to fmu input objects

        :param u: optimization vectors from optimizer
        :type u: numpy.array or list
        """
        nu = len(u)
        ni = len(self.fmu_input_names)

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

    def ga(self):

        pass 

if __name__ == "__main__":
    model = load_fmu("SingleZoneFCU.fmu")
    states_names = model.get_states_list()
    states = [float(model.get(state)) for state in states_names]
    PH = 8
    CH = 1
    dt = 900.
    ts = 201*24*3600.

    price = [0.02987, 0.02987, 0.02987, 0.02987,
             0.02987, 0.02987, 0.04667, 0.04667,
             0.04667, 0.04667, 0.04667, 0.04667,
             0.15877, 0.15877, 0.15877, 0.15877,
             0.15877, 0.15877, 0.15877, 0.04667,
             0.04667, 0.04667, 0.02987, 0.02987]

    measurement_names = ['uFan', 'PTot', 'TRoo']
    control_names = ['uFan']
    mpc = PerfectMPC(PH = PH,
                    CH = CH,
                    time = ts,
                    dt = dt,
                    fmu_model = model, 
                    measurement_names = measurement_names,
                    control_names = control_names,
                    price = price)

    # test states
    mpc.initialize_fmu()
    states = mpc.get_fmu_states()
    
    # Test objective 
    mpc.u_prev = [1]
    u = [0.5]*PH
    mpc.set_time(ts)
    mpc.set_mpc_states(states)

    input = mpc._transfer_inputs(u, piecewise_constant=True)
    states_next, output = mpc.simulate(ts, ts+7200., input=input)

    t = np.arange(ts, ts+7200., 60.)
    output = mpc._interpolate(output,t)
    output = mpc._sample(output, dt)

    energy_cost = mpc._calculate_energy_cost(output)
    temp_violations = mpc._calculate_temperature_violation(output)
    action_changes = mpc._calculate_action_changes(u)

    print(energy_cost)
    print(temp_violations)
    print(action_changes)

    #
    mpc.reset_fmu()
    #mpc.initialize_fmu()
    states = mpc.get_fmu_states()
    print(states)
    print(len(states))
    mpc.set_time(ts)
    mpc.set_mpc_states(states)
    print(mpc.objective(u))
    #
    mpc.optimize()

    """
    # test optimizer
    # Define the objective function
    # This function has a local minimum f = 48.98 at x = np.array([11.41, -0.8968])
    # and a global minimum f = 0 at x = np.array([5.0, 4.0])
    def freudenstein_roth(x):
        r1 = -13.0 + x[0] + ((5.0 - x[1]) * x[1] - 2.0) * x[1]
        r2 = -29.0 + x[0] + ((1.0 + x[1]) * x[1] - 14.0) * x[1]
        return r1 ** 2 + r2 ** 2

    # Define the starting point
    x0 = np.array([5.0, -20.0])

    # Define bounds (required for global optimization)
    lower = np.array([-30.0, -30.0])
    upper = np.array([30.0, 30.0])

    print("First run - search for local minimum only")
    print("")
    soln = pybobyqa.solve(freudenstein_roth, x0, maxfun=500, bounds=(lower, upper), seek_global_minimum=True)
    print(soln)
    """
