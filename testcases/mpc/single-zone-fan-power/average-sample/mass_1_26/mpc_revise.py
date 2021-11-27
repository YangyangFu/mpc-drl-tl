# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

import numpy as np
# ipopt
#from pyomo.environ import *

import casadi as ca
import model
import joblib

class PowerCallback(ca.Callback):
    def __init__(self,name, PH, n, power_model, opts={}):
        ca.Callback.__init__(self)
        self.PH = PH # prediction horizon
        self.n = n # number of optimization variables at each step
        self.power_model = power_model
        self.construct(name,opts)

    # Number of inputs and outputs   
    def get_n_in(self): return 1
    def get_n_out(self): return 1

    # Array of inputs and outputs
    def get_sparsity_in(self,i):
        return ca.Sparsity.dense(self.n*self.PH,1)
    def get_sparsity_out(self,i):
        return ca.Sparsity.dense(self.PH,1)

    # Initialize the object
    def init(self):
        print('initializing object')

    def power_polynomial(self,mz):
        params = self.power_model
        f_P = model.FanPower(n=len(params['alpha']))
        f_P.params = params
        PFan = f_P.predict(mz)

        return PFan

    def eval(self,arg):
        """evaluate objective

        """
        # get control inputs: U = {u(t+1), u(t+2), ... u(t+PH)}
        #                      u = [mz, \epsilon]
        U = arg[0]

        # loop over the prediction horizon
        i = 0
        P_pred_ph = []

        while i < self.PH:
            # get u for current step 
            u = U[i*self.n:self.n*(i+1)]
            mz = u[0]*0.75 # control inputs

            ### ====================================================================
            ###            Power Predictor
            ### =====================================================================
            # predict total power at current step
            P_pred = self.power_polynomial(mz) 
            # Save step-wise power prediction 
            P_pred_ph.append(P_pred) # save all step-wise power for cost calculation
            ### ===========================================================
            ###      Update for next step
            ### ==========================================================
            # update clock
            i += 1

        return [ca.DM(P_pred_ph)]

class ZoneTemperatureCallback(ca.Callback):
    def __init__(self,name, PH, n, zone_model, states, predictor, opts={}):
        ca.Callback.__init__(self)
        self.PH = PH
        self.n = n
        self.zone_model = zone_model # params={'alpha':{}}
        self.states = states
        self.predictor = predictor
        self.instantiate_zone_arx_model()
        self.construct(name,opts)

    # Number of inputs and outputs   
    def get_n_in(self): return 1
    def get_n_out(self): return 1

    # Array of inputs and outputs
    def get_sparsity_in(self,i):
        return ca.Sparsity.dense(self.n*self.PH,1)
    def get_sparsity_out(self,i):
        return ca.Sparsity.dense(self.PH,1)

    # Initialize the object
    def init(self):
        print('initializing object')

    def instantiate_zone_arx_model(self):
        """Instantiate a zone arx model
        """
        self.Lz = len(self.zone_model['alpha'])
        self.Lo = len(self.zone_model['beta'])
        self.f_Tz = model.Zone(Lz=self.Lz, Lo=self.Lo)
        self.f_Tz.params = self.zone_model

    def zone_arx(self,Tz_his_meas, To_his_meas, mz):
        
        Tsa = 13
        return self.f_Tz.model(Tz_his_meas, To_his_meas, mz, Tsa)

    def eval(self,arg):
        """evaluate temperature

        """
        # get control inputs: U = {u(t+1), u(t+2), ... u(t+PH)}
        #                      u = [mz, \epsilon]
        U = arg[0]

        # loop over the prediction horizon
        i = 0
        Tz_pred_ph = []

        # get states at current step t
        Tz_his_meas = self.states['Tz_his_meas'] # [Tz(t-l), ..., Tz(t-1), Tz(t)] 
        To_his_meas = self.states['To_his_meas'] # [Tz(t-l), ..., Tz(t-1), Tz(t)] 
        Tz_his_pred = self.states['Tz_his_pred'] # [Tz(t-l), ..., Tz(t-1), Tz(t)] 

        # get autocorrection term 
        e = self.f_Tz.autocorrection(Tz_his_meas, Tz_his_pred)

        while i < self.PH:
            # get u for current step 
            u = U[i*self.n:self.n*(i+1)]
            mz = u[0]*0.75 # control inputs

            ### ====================================================================
            ###           Zone temperature prediction
            ### =====================================================================
            # predict total power at current step
            Tz_pred = self.zone_arx(Tz_his_meas, To_his_meas, mz) + e

            ### ===========================================================
            ###      Update for next step
            ### ==========================================================
            # update states for zone recursive regression model
            # update future measurement
            # future zone temperature measurements
            Tz_his_meas = LIFO(Tz_his_meas,Tz_pred)

            # future oa temperature measurement
            To_pred = self.predictor['Toa'][i]
            To_his_meas = LIFO(To_his_meas,To_pred)

            # save step-wise temperature
            Tz_pred_ph.append(Tz_pred) # save all step-wise power for cost calculation

            # update clock
            i += 1

        # return predicted zone temperature over prediction horizon

        return [Tz_pred_ph]


def LIFO(array,x):
    # lAST IN FIRST OUT:
    a = np.append(array,x)

    return list(a[1:])

class mpc_case():
    def __init__(self,PH,CH,time,dt,zone_model, power_model,measurement,states,predictor):

        self.PH=PH # prediction horizon
        self.CH=CH # control horizon
        self.dt = dt # time step
        self.time = time # current time index

        #self.parameters_zone = parameters_zone # parameters for zone dynamic mpc model, dictionary
        #self.parameters_power = parameters_power # parameters for system power dynamic mpc model, dictionary
        self.measurement = measurement # measurement at current time step, dictionary
        self.predictor = predictor # price and outdoor air temperature for the future horizons
        
        ## load MPC models
        # should be decided beforehand: if ARX, {'alpha':[], 'beta':[], 'gamma':[]}; if ANN, {'model_name':'xx.pkl'}
        self.zone_model = zone_model if 'model_name' not in zone_model else joblib.load(zone_model['model_name'])
        self.power_model = power_model if 'model_name' not in power_model else joblib.load(power_model['model_name'])

        self.states = states # dictionary

        # some building control settings
        self.number_zone = 1
        self.occ_start = 7 # occupancy starts
        self.occ_end = 19 # occupancy ends

        # some mpc settings
        self.n = 2 # number of control variable for each step
        self.w = [1., 1.] # weights between energy cost and temperature violation
        self.u_lb = [0.]*self.n
        self.u_ub = [1.,0.1]
        # initialize optimiztion
        self.u_start = self.u_lb*self.PH
        #self.optimization_model=self.get_optimization_model() # pyomo object
        self.optimum= {}

        # save internal power and temp predictor casadi function for extra calls
        self._P = None
        self._Tz = None

    def optimize(self):
        """MPC optimization problem in casadi interface
        """

        # instantiate power function
        P = PowerCallback('P', 
                            PH = self.PH,
                            n = self.n,
                            power_model = self.power_model,
                            opts={"enable_fd":True})   
             
        # instantiate nonlinear constraints zone temperature
        Tz = ZoneTemperatureCallback('Tz',
                                    PH = self.PH, 
                                    n = self.n,
                                    zone_model = self.zone_model, 
                                    states = self.states, 
                                    predictor =self.predictor, 
                                    opts={"enable_fd":True})
        # define casadi variables
        u = ca.MX.sym("U",self.n*self.PH)

        # define objective function
        power_ph=P(u)
        price_ph = self.predictor['price']
        fval = []
        for k in range(self.PH):
            fo = self.w[0]*power_ph[k]*price_ph[k]*self.dt/3600./1000 + self.w[1]*u[self.n*k+1]**2
            fval.append(fo)
        fval_sum = ca.sum1(ca.vertcat(*fval))

        obj=ca.Function('fval',[u],[fval_sum]) # this is the ultimate objective function
        f = obj(u) # get the objective value

        # define constraint function
        Tz_pred = Tz(u) # predicted T of size PH

        ### define nonlinear temperature constraints
        # zone temperature bounds - need check with the high-fidelty model
        T_upper = np.array([30.0 for i in range(24)])
        T_upper[self.occ_start:self.occ_end] = 26.0
        T_lower = np.array([12.0 for i in range(24)])
        T_lower[self.occ_start:self.occ_end] = 22.0

        # get overshoot and undershoot for each step
        # current time step
        time = self.time

        g = []
        lbg = []
        ubg = []
        for k in range(self.PH):
            # future time        
            t = int(time+k*self.dt)
            t = int((t % 86400)/3600)  # hour index 0~23
            
            # inequality constraints
            eps = u[self.n*k+1]
            g += [Tz_pred[k]+eps, Tz_pred[k]-eps]
            # get upper and lower T bound
            lbg += [T_lower[t], 0.]
            ubg += [ca.inf, T_upper[t]]

        # formulate an optimziation problem using nlp solver
        prob = {'f':f, 'x': u, 'g': ca.vertcat(*g)}
        opts = {}
        nlp_optimize = ca.nlpsol('solver', 'ipopt', prob, opts)

        # set initial guess
        u0 = self.u_start
        print(u0)
        # set lower upper bound for u
        lbx = self.u_lb*self.PH
        ubx = self.u_ub*self.PH

        # solve the optimization problem
        solution = nlp_optimize(x0=u0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        
        # get and save the results
        u_opt = solution['x']
        f_opt = solution['f']
        self.optimum['objective'] = f_opt
        self.optimum['variable'] = u_opt

        print(solution)
        
        # save the function for external calls
        self._P = P
        self._Tz = Tz

        return self.optimum
        
    def set_time(self, time):
        
        self.time = time

    def set_mpc_model_parameters(self):
        pass

    def set_measurement(self,measurement):
        """Set measurement at time t

        :param measurement: system measurement at time t
        :type measurement: pandas DataFrame
        """
        self.measurement = measurement
    
    def set_states(self,states):
        """Set states for current time step t

        :param states: values of states at t
        :type states: dict

            for example:

            states = {'Tz_his_t':[24,23,24,23]}

        """
        self.states = states
    
    def set_predictor(self, predictor):
        """Set predictor values for current time step t

        :param predictor: values of predictors from t to t+PH
        :type predictor: dict

            for example:

            predictor = {'energy_price':[1,3,4,5,6,7,8]}

        """
        self.predictor = predictor

    def get_u_start(self,optimum_prev):
        fut = optimum_prev[self.n:]
        start = np.append(fut, [0.1,0.])
        return start

    def set_u_start(self,prev):
        """Set start value for design variables using previous optimization results
        """
        start = self.get_u_start(prev)
        self.u_start = start
