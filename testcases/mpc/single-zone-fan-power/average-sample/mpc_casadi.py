# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

import numpy as np
# ipopt
#from pyomo.environ import *

import casadi as ca
import model
import joblib

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
        self.P_his_t = []
        self.Tz_his_t = []

        self.number_zone = 1
        self.occ_start = 6 # occupancy starts
        self.occ_end = 19 # occupancy ends

        # initialize optimiztion
        #self.optimization_model=self.get_optimization_model() # pyomo object
        self.optimum= {}
        self.u_start = [0.5]*PH
        self.u_lb = [0.0]*PH
        self.u_ub = [1.0]*PH


    def obj(self,u_ph):
        #u_ph = u_ph.inputs
        # MPC model predictor settings
        alpha_power = np.array(self.parameters_power['alpha'])
        #beta_power = np.array(self.parameters_power['beta'])
        #gamma_power = np.array(self.parameters_power['gamma'])

        alpha_zone = np.array(self.parameters_zone['alpha'])
        beta_zone = np.array(self.parameters_zone['beta'])
        gamma_zone = np.array(self.parameters_zone['gamma'])  
        l = 4

        # zone temperature bounds - need check with the high-fidelty model
        T_upper = np.array([30.0 for i in range(24)])
        T_upper[self.occ_start:self.occ_end] = 26.0
        T_lower = np.array([12.0 for i in range(24)])
        T_lower[self.occ_start:self.occ_end] = 22.0

        overshoot = []
        undershoot = []

        # loop over the prediction horizon
        i = 0
        P_pred_ph = []
        Tz_pred_ph = [] 

        # get current state - has to copy original states due to mutable object list and dictionary
        P_his = self.states['P_his_t'][:] # current state of historical power [P(t) to P(t-l)]   
        Tz_his = self.states['Tz_his_t'][:] # current zone temperatur states for zone temperture prediction model  
        
        # initialize the cost function
        penalty = []  # temperature violation penalty for each zone
        alpha_up = 0.01
        alpha_low = 0.01
        time = self.time

        while i < self.PH:

            mz = u_ph[i]*0.75 # control inputs
            Toa = self.predictor['Toa'][i] # predicted outdoor air temperature

            ### ====================================================================
            ###            Power Predictor
            ### =====================================================================
            # predict total power at current step
            #P_pred = self.total_power(alpha_power, beta_power, gamma_power, l, P_his, mz, Toa) 
            P_pred = self.total_power(alpha_power, mz) 
            # update historical power measurement for next prediction
            # reverst the historical data to enable FILO: [t, t-1, ...,t-l]
            P_his = self.FILO(P_his,P_pred)

            # Save step-wise power
            P_pred_ph.append(P_pred) # save all step-wise power for cost calculation

            ### =================================================================
            ###       Zone Temperatre Predicction
            ### ============================================================
            Ts = 273.15+13 # used as constant in this model
            #Tz_pred = self.zone_temperature(alpha_zone, beta_zone, gamma_zone, l, Tz_his, mz, Ts, Toa)
            Tz_pred = self.zone_temperature(Tz_his, mz, Toa)
            # update historical zone temperature measurement for next prediction
            # reverse the historical data to enable FILO
            Tz_his = self.FILO(Tz_his,Tz_pred)

            # save step-wise temperature
            Tz_pred_ph.append(Tz_pred) # save all step-wise power for cost calculation

            # get overshoot and undershoot for each step
            # current time step
            t = int(time+i*self.dt)
            t = int((t % 86400)/3600)  # hour index 0~23

            # this is to be revised for multi-zone 
            for k in range(self.number_zone):
                overshoot.append(np.array([float((Tz_pred -273.15) - T_upper[t]), 0.0]).max())
                undershoot.append(np.array([float(T_lower[t] - (Tz_pred-273.15)), 0.0]).max())
            
            ### ===========================================================
            ###      Update for next step
            ### ==========================================================
            # update clock

            i+=1

        # calculate total cost based on predicted energy prices
        price_ph = self.predictor['price']

        # zone temperature bounds penalty
        penalty = alpha_up*sum(np.array(overshoot)) + alpha_low*sum(np.array(undershoot))

        ener_cost = float(np.sum(np.array(price_ph)*np.array(P_pred_ph)))*self.dt/3600./1000. 

        #print mz, np.array(Tz_pred_ph)-273.15, ener_cost, penalty

        # objective for a minimization problem
        f = ener_cost + penalty
        #f = ener_cost
        #f=penalty
        # constraints - unconstrained
        #g=[0.0]*self.PH
        #for i in range(self.PH):
        #    g[i] = 0.1 - u_ph[i]


        # simulation status
        #fail = 0
        #print f, g
        return f

    def optimize(self):
        """
        MPC optimizer: call optimizer to solve a minimization problem
        
        """
       
        self.optimum={}

        # get optimization model
        model = self.get_optimization_model()

        # solve optimization
        xtol=1e-3
        ftol=1e-6
        solver='de' # GLP
        #solver = "interalg" # GLP
        #solver='ralg' # NLP
        #solver='ipopt'# NLP
        r = model.solve(solver,xtol=xtol,ftol=ftol)
        x_opt, f_opt = r.xf, r.ff
        d1 = r.evals
        d2 = r.elapsed
        nbr_iters = d1['iter']
        nbr_fevals = d1['f']
        solve_time = d2['solver_time']
        

        self.optimum['objective'] = f_opt
        self.optimum['variable'] = x_opt

        disp = True
        if disp:
            print (' ')
            print ('Solver: OpenOpt solver ' + solver)
            print (' ')
            print ('Number of iterations: ' + str(nbr_iters))
            print ('Number of function evaluations: ' + str(nbr_fevals))
            print (' ')
            print ('Execution time: ' + str(solve_time))
            print (' ')

        return self.optimum

    def openopt_model_glp(self):
        """This is to formulate a global optimization problem
        """
        # create oovar of size PH
        #u = oovar(size=self.PH)
        # might not work becasue u is a oovar object, not list
        objective = lambda u: self.obj([u[i] for i in range(self.PH)])
        #startPoint = 0.5*np.array(self.PH)

        # bounds
        lb = self.u_lb
        ub = self.u_ub

        return GLP(objective, lb=lb, ub=ub, maxIter = 5e5)

    def openopt_model_nlp(self):
        """This is to formulate a nolinear programming problem in openopt: minimization
        """
        objective = lambda u: self.obj([u[i] for i in range(self.PH)])
        # objective gradient - optional
        df = None
        # start point
        start = self.u_start
        # constraints if any
        c = None # nonlinear inequality
        dc = None # derivative of c
        h = None # nonlinear equality
        dh = None # derivative of h
        A = None # linear inequality
        b = None
        Aeq = None # linear equality
        beq = None # linear equality

        # bounds
        lb = self.u_lb
        ub = self.u_ub

        # tolerance control
        # required constraints tolerance, default for NLP is 1e-6
        contol = 1e-6

        # If you use solver algencan, NB! - it ignores xtol and ftol; using maxTime, maxCPUTime, maxIter, maxFunEvals, fEnough is recommended.
        # Note that in algencan gtol means norm of projected gradient of  the Augmented Lagrangian
        # so it should be something like 1e-3...1e-5
        gtol = 1e-6 # (default gtol = 1e-6)

        # see https://github.com/troyshu/openopt/blob/d15e81999ef09d3c9c8cb2fbb83388e9d3f97a98/openopt/oo.py#L390.
        return NLP(objective, start, df=df,  c=c,  dc=dc, h=h,  dh=dh,  A=A,  b=b,  Aeq=Aeq,  beq=beq,  
        lb=lb, ub=ub, gtol=gtol, contol=contol, maxIter = 50000, maxFunEvals = 20000, name = 'NLP for: '+str(self.time))

    def get_optimization_model(self):
        return self.openopt_model_glp()
        #return self.openopt_model_nlp()

    def zone_temperature(self,Tz_his, mz, Toa):
        ann = self.zone_model
        x=list(Tz_his)
        x.append(mz)
        x.append(Toa)
        x = np.array(x).reshape(1,-1)
        y=float(ann.predict(x))
        
        return np.maximum(273.15+14,np.minimum(y,273.15+35))

    def total_power(self,alpha, mz):
        """Predicte power at next step

        :param alpha: coefficients from curve-fitting
        :type alpha: np array (l,)
        :param beta: coefficient from curve-fitting
        :type beta: scalor
        :param gamma: coefficient from curve-fitting
        :type gamma: scalor
        :param l: historical step
        :type l: scalor
        :param Tz_his: historical zone temperature array
        :type Tz_his: np array (l,)
        :param mz: zone air mass flowrate at time t
        :type mz: scalor
        :param Ts: discharge air temperaure at time t
        :type Ts: scalor
        :param Toa: outdoor air dry bulb temperature at time t
        :type Toa: scalor

        :return: predicted zone temperature at time t
        :rtype: scalor
        """
        # check dimensions

        alpha=np.array(alpha).reshape(-1)
        #beta=np.array(beta).reshape(-1)

        P = alpha[0]+alpha[1]*mz+alpha[2]*mz**2 #+ beta[0]+ beta[1]*Toa+beta[2]*Toa**2

        return float(abs(P))

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
        fut = optimum_prev[1:]
        start = np.append(fut,(self.u_lb[-1]+self.u_ub[-1])/2.)
        return start

    def set_u_start(self,prev):
        """Set start value for design variables using previous optimization results
        """
        start = self.get_u_start(prev)
        self.u_start = start

class ObjectiveCallback(ca.Callback):
    def __init__(self,name,time, PH, w, power_model, states, opts={}):
        ca.Callback.__init__(self)
        self.time = time
        self.PH = PH
        self.params_power = power_model
        self.w = w # weights for energy cost and temperature violation term [w1, w2]
        self.construct(name,opts)

    # Number of inputs and outputs   
    def get_n_in(self): return 1
    def get_n_out(self): return 1

    # Array of inputs and outputs
    def get_sparsity_in(self,i):
        return ca.Sparsity.dense(2*self.PH,1)
    def get_sparsity_out(self,i):
        return ca.Sparsity.dense(1,1)

    # Initialize the object
    def init(self):
        print('initializing object')

    def power_polynomial(self,mz):
        params = self.power_model
        f_P = model.FanPower(n=len(params['alpha']))
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
            u = U[i*2:2*i+2]
            mz = u[0]*0.75 # control inputs
            eps = u[1] # temperature slack

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

        # energy cost: calculate total cost based on predicted energy prices
        price_ph = self.predictor['price']
        energy_cost = float(np.sum(np.array(price_ph)*np.array(P_pred_ph)))*self.dt/3600./1000. 

        # zone temperature bounds penalty
        penalty = eps**2

        # objective for a minimization problem
        f = self.w[0]*energy_cost + self.w[1]*penalty

        return [f]


class ZoneTemperatureCallback(ca.Callback):
    def __init__(self,name,time, PH, zone_model, states, predictor, opts={}):
        ca.Callback.__init__(self)
        self.time = time
        self.PH = PH
        self.params_zone = zone_model # params={'alpha':{}}
        self.states = states
        self.predictor = predictor
        self.instantiate_zone_model()
        self.construct(name,opts)

    # Number of inputs and outputs   
    def get_n_in(self): return 1
    def get_n_out(self): return 1

    # Array of inputs and outputs
    def get_sparsity_in(self,i):
        return ca.Sparsity.dense(2*self.PH,1)
    def get_sparsity_out(self,i):
        return ca.Sparsity.dense(self.PH,1)

    # Initialize the object
    def init(self):
        print('initializing object')

    def instantiate_zone_model(self):
        """Instantiate a zone model
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
            u = U[i*2:2*i+2]
            mz = u[0]*0.75 # control inputs

            ### ====================================================================
            ###           Zone temperature prediction
            ### =====================================================================
            # predict total power at current step
            Tz_pred = self.zone_arx(Tz_his_meas, To_his_meas, mz) + e
            # Save step-wise power prediction 
            Tz_pred_ph.append(Tz_pred) # save all step-wise temperature for output

            ### ===========================================================
            ###      Update for next step
            ### ==========================================================
            # update states for zone recursive regression model
            # update future measurement
            # future zone temperature measurements
            Tz_his_meas = FILO(Tz_his_meas,Tz_pred)
            # future oa temperature measurement
            To_pred = self.predictor['Toa'][i]
            To_his_meas = FILO(To_his_meas,To_pred)

            # save step-wise temperature
            Tz_pred_ph.append(Tz_pred) # save all step-wise power for cost calculation

            # update clock
            i += 1

        # return predicted zone temperature over prediction horizon
        return Tz_pred_ph


def FILO(lis,x):
    # FIRST IN LAST OUT:
    lis.pop() # remove the last element
    lis.reverse()
    lis.append(x)
    lis.reverse()

    return lis
