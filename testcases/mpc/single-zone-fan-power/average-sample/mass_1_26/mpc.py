# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

import numpy as np
# ipopt
#from pyomo.environ import *

import casadi as ca
import model
import joblib
from copy import copy 

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
        self.w = [1000., 10, 1.] # weights between energy cost and temperature violation and actions violations
        self.u_lb = [0.]*self.n
        self.u_ub = [1.,0.1]
        # initialize optimiztion
        self.u_start = [self.u_lb[i%self.n]+0.01*float(np.random.rand(1)) for i in range(self.n*self.PH)]
        #self.optimization_model=self.get_optimization_model() # pyomo object
        self.optimum={}
        self.x_opt_0 = self.u_lb
        
        # save internal power and temp predictor casadi function for extra calls
        self._P = None
        self._Tz = None
        self._autoerror= 0

    def optimize(self):
        """MPC optimization problem in casadi interface
        """
        # unwrap system states and predictors
        # get states at current step t
        Tz_his_meas = self.states['Tz_his_meas'] # [Tz(t-l), ..., Tz(t-1), Tz(t)] 
        To_his_meas = self.states['To_his_meas'] # [Tz(t-l), ..., Tz(t-1), Tz(t)] 
        Tz_his_pred = self.states['Tz_his_pred'] # [Tz(t-l), ..., Tz(t-1), Tz(t)] 
        # get price predictor
        price_ph = self.predictor['price']
        To_pred_ph = self.predictor['Toa']

        # define casadi variables
        u = ca.MX.sym("U",self.n*self.PH)

        # define objective function
        n_Tz_his = len(Tz_his_meas)
        autoerror = 0
        for i in range(n_Tz_his):
            autoerror += (Tz_his_meas[i]-Tz_his_pred[i])/n_Tz_his
        
        fval = []
        Tz_pred_ph = []
        P_pred_ph = []
        Tz_his_meas_k = [Tz_meas for Tz_meas in Tz_his_meas]
        To_his_meas_k = [To_meas for To_meas in To_his_meas]
        mFan_nominal = 0.4 #kg/s
        u_prev = self.x_opt_0

        for k in range(self.PH):
            # predict power 
            P_pred_ph.append(self.predict_power(u[self.n*k]*mFan_nominal)) 
            # predict future zone temperature
            Tz_pred_ph.append(self.predict_zone_temp(
                Tz_his_meas_k, To_his_meas_k, u[self.n*k]*mFan_nominal, 13, autoerror))

            # update historical temperature over the PH
            Tz_his_meas_k.append(Tz_pred_ph[k])
            Tz_his_meas_k = Tz_his_meas_k[1:]
            To_his_meas_k.append(To_pred_ph[k])
            To_his_meas_k = To_his_meas_k[1:]

            # control actions
            normalizer = [1/(self.u_ub[i]-self.u_lb[i]) for i in range(self.n)]
            u_k = u[k*self.n:(k+1)*self.n] 
            #du_k=[u_k[i]-u_prev[i] for i in range(self.n)]
            du_k = u_k - u_prev
            u_prev = u_k # update previous actions in PH
            du_k_normalized = [normalizer[i]*du_k[i] for i in range(self.n-1)]#ca.dot(normalizer[:-1], du_k[:-1])
            du_k_nom2 = ca.sumsqr(ca.vertcat(*du_k_normalized))

            # get objective function
            fo = self.w[0]*P_pred_ph[k]*price_ph[k]*self.dt/3600./1000 + self.w[1]*u[self.n*k+1]**2 + self.w[2]*du_k_nom2
            fval.append(fo)

        fval_sum = ca.sum1(ca.vertcat(*fval))

        obj=ca.Function('fval',[u],[fval_sum]) # this is the ultimate objective function
        f = obj(u) # get the objective value

        ### define nonlinear temperature constraints
        # zone temperature bounds - need check with the high-fidelty model
        T_upper = np.array([30.0 for i in range(24)])
        T_upper[self.occ_start:self.occ_end] = 26.0
        T_lower = np.array([12.0 for i in range(24)])
        T_lower[self.occ_start:self.occ_end] = 22.0

        # get overshoot and undershoot for each step
        # current time step
        time = self.time
        h_time = int(time % 86400/3600)
        
        if h_time == 11:
            print("=======================\ntime:" + str(h_time))
            print(time)
            print(price_ph)
            input("Press Enter to continue")
        g = []
        lbg = []
        ubg = []
        for k in range(self.PH):
            # future time        
            t = int(time+k*self.dt)
            t = int((t % 86400)/3600)  # hour index 0~23
            
            # inequality constraints
            eps = u[self.n*k+1]
            g += [Tz_pred_ph[k]+eps, Tz_pred_ph[k]-eps]
            # get upper and lower T bound
            lbg += [T_lower[t], 0.]
            ubg += [ca.inf, T_upper[t]]

        # formulate an optimziation problem using nlp solver
        prob = {'f':f, 'x': u, 'g': ca.vertcat(*g)}
        opts = {}
        nlp_optimize = ca.nlpsol('solver', 'ipopt', prob, opts)

        # set initial guess
        u0 = self.u_start
        print("Initial Guess:", u0)
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
        # save optimum results for next step
        self.x_opt_0 = u_opt
        print(solution)
        
        # save the function for external calls
        self._autoerror = autoerror
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

    def set_u_prev(self, u_prev):
        """
        set control actions from previous step

        :param u_prev: previous control action vector
        :type u_prev: list
        """
        self.x_opt_0 = u_prev

    def predict_power(self, mz):
        """
        Power model

        :param params: _description_
        :type params: _type_
        """
        params = self.power_model['alpha']
        n = len(params)
        P=0
        for i in range(n):
            P += params[i]*mz**i
        return P

    def predict_zone_temp(self, Tz_his_meas, To_his_meas, mz, Tsa, autoerror):
        alpha = self.zone_model['alpha']  # list
        beta = self.zone_model['beta'] # list
        gamma = self.zone_model['gamma'] # list
        
        n_alpha = len(alpha)
        n_beta = len(beta)

        Tz_pred = ca.sum1(ca.vertcat(*[alpha[i]*Tz_his_meas[i] for i in range(n_alpha)])) \
            + ca.sum1(ca.vertcat(*[beta[i]*To_his_meas[i] for i in range(n_beta)])) \
            + ca.MX(gamma*mz*(Tsa-Tz_his_meas[0])) \
            + ca.MX(autoerror)

        return Tz_pred
