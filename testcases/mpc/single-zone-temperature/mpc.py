'''

'''
import numpy as np
# ipopt
#from pyomo.environ import *

from pyjmi.optimization import dfo
from FuncDesigner import *
from openopt import NSP
from openopt import NLP
from openopt import GLP 
from sklearn.externals import joblib

class mpc_case():
    def __init__(self,PH,CH,time,dt,parameters_zone, parameters_power,measurement,states,predictor):

        self.PH=PH # prediction horizon
        self.CH=CH # control horizon
        self.dt = dt # time step
        self.time = time # current time index

        self.parameters_zone = parameters_zone # parameters for zone dynamic mpc model, dictionary
        self.parameters_power = parameters_power # parameters for system power dynamic mpc model, dictionary
        self.measurement = measurement # measurement at current time step, dictionary
        self.predictor = predictor # price and outdoor air temperature for the future horizons

        self.zone_model = joblib.load('TZoneANN.pkl')
        self.power_model = joblib.load('powerANN.pkl')

        self.states = states # dictionary
        self.P_his_t = []
        self.Tz_his_t = []

        self.number_zone = 1
        self.occ_start = 6 # occupancy starts
        self.occ_end = 19 # occupancy ends

        # initialize optimiztion
        #self.optimization_model=self.get_optimization_model() # pyomo object
        self.optimum= {}
        self.u_start = [273.15+24]*PH
        self.u_lb = [273.15+22]*PH
        self.u_ub = [273.15+26]*PH


    def obj(self,u_ph):
        # MPC model predictor settings
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

            ui = u_ph[i] # control inputs
            Toa = self.predictor['Toa'][i] # predicted outdoor air temperature

            ### =================================================================
            ###       Zone Temperatre Predicction
            ### ============================================================
            Ts = 273.15+13 # used as constant in this model
            Tz_pred = self.zone_temperature(Tz_his, ui, Toa)


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

            ### ====================================================================
            ###            Power Predictor
            ### =====================================================================
            # predict total power at current step
            #P_pred = self.total_power(alpha_power, mz) 
            P_pred = self.cool_power(P_his, Tz_his, Tz_pred, ui, Toa, t)
            # Save step-wise power
            P_pred_ph.append(P_pred) # save all step-wise power for cost calculation

            # update historical zone temperature measurement for next prediction
            # reverse the historical data to enable FILO
            Tz_his = self.FILO(Tz_his,Tz_pred)
            # update historical power measurement for next prediction
            # reverst the historical data to enable FILO: [t, t-1, ...,t-l]
            P_his = self.FILO(P_his,P_pred)

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
            print ' '
            print 'Solver: OpenOpt solver ' + solver
            print ' '
            print 'Number of iterations: ' + str(nbr_iters)
            print 'Number of function evaluations: ' + str(nbr_fevals)
            print ' '
            print 'Execution time: ' + str(solve_time)
            print ' '

        return self.optimum

    def openopt_model_glp(self):
        """This is to formulate a global optimization problem
        """
        # create oovar of size PH
        #u = oovar(size=self.PH)
        # might not work becasue u is a oovar object, not list
        objective = lambda u: self.obj([u[i] for i in range(self.PH)])
        #startPoint = 0.5*np.array(self.PH)

        # bounds - adaptive for occupied and unoccupied hours
        lb, ub = self.moving_constraints()

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

        # bounds - adaptive for occupied and unoccupied hours
        lb, ub = self.moving_constraints()

        # tolerance control
        # required constraints tolerance, default for NLP is 1e-6
        contol = 1e-6

        # If you use solver algencan, NB! - it ignores xtol and ftol; using maxTime, maxCPUTime, maxIter, maxFunEvals, fEnough is recommended.
        # Note that in algencan gtol means norm of projected gradient of  the Augmented Lagrangian
        # so it should be something like 1e-3...1e-5
        gtol = 1e-6 # (default gtol = 1e-6)

        # see https://github.com/troyshu/openopt/blob/d15e81999ef09d3c9c8cb2fbb83388e9d3f97a98/openopt/oo.py#L390.
        return NLP(objective, start, df=df,  c=c,  dc=dc, h=h,  dh=dh,  A=A,  b=b,  Aeq=Aeq,  beq=beq,  
        lb=lb, ub=ub, gtol=gtol, contol=contol, maxIter = 10000, maxFunEvals = 1e4, name = 'NLP for: '+str(self.time))

    def get_optimization_model(self):
        #return self.openopt_model_glp()
        return self.openopt_model_nlp()

    def FILO(self,lis,x):
        lis.pop() # remove the last element
        lis.reverse()
        lis.append(x)
        lis.reverse()

        return lis

    def moving_constraints(self):
        t = self.time
        h = int((t % 86400)/3600)  # hour index 0~23
        PH = self.PH
        dt = self.dt 

        # zone temperature bounds - need check with the high-fidelty model
        T_upper = np.array([30.0 for i in range(24)])
        T_upper[self.occ_start:self.occ_end] = 26.0
        T_lower = np.array([12.0 for i in range(24)])
        T_lower[self.occ_start:self.occ_end] = 22.0

        # 
        self.u_lb = [T_lower[int((t+ph*dt) % 86400 / 3600)]+273.15 for ph in range(PH)]
        self.u_ub = [T_upper[int((t+ph*dt) % 86400 / 3600)]+273.15 for ph in range(PH)]

        return self.u_lb, self.u_ub
    def zone_temperature(self,Tz_his, mz, Toa):
        ann = self.zone_model
        x=list(Tz_his)
        x.append(mz)
        x.append(Toa)
        x = np.array(x).reshape(1,-1)
        y=float(ann.predict(x))
        
        return np.maximum(273.15+14,np.minimum(y,273.15+35))

    def cool_power(self,P_his, Tz_his, Tz_pred, Tz_set, Toa,h):
        ann = self.power_model
        x = list(P_his)
        x += list(Tz_his)
        x.append(Tz_pred)
        x.append(Tz_set)
        x.append(Toa)
        x.append(h)
        x = np.array(x).reshape(1,-1)

        y=float(ann.predict(x))
        
        return y


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
