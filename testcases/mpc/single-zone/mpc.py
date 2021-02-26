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

        self.states = states # dictionary
        self.P_his_t = []
        self.Tz_his_t = []

        self.number_zone = 1
        self.occ_start = 6 # occupancy starts
        self.occ_end = 19 # occupancy ends

        # initialize optimiztion
        #self.optimization_model=self.get_optimization_model() # pyomo object
        self.optimum={}


    def obj(self,u_ph):
        #u_ph = u_ph.inputs
        # MPC model predictor settings
        alpha_power = np.array(self.parameters_power['alpha'])
        beta_power = np.array(self.parameters_power['beta'])
        gamma_power = np.array(self.parameters_power['gamma'])

        alpha_zone = np.array(self.parameters_zone['alpha'])
        beta_zone = np.array(self.parameters_zone['beta'])
        gamma_zone = np.array(self.parameters_zone['gamma'])  
        l = 4

        # zone temperature bounds - need check with the high-fidelty model
        T_upper = np.array([30.0 for i in range(24)])
        T_upper[self.occ_start:self.occ_end] = 25.0
        T_lower = np.array([18.0 for i in range(24)])
        T_lower[self.occ_start:self.occ_end] = 23.0

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
        alpha_up = 200.0
        alpha_low = 200.0
        time = self.time

        while i < self.PH:

            mz = u_ph[i]*0.75 # control inputs
            Toa = self.predictor['Toa'][i] # predicted outdoor air temperature

            ### ====================================================================
            ###            Power Predictor
            ### =====================================================================
            # predict total power at current step
            P_pred = self.total_power(alpha_power, beta_power, gamma_power, l, P_his, mz, Toa) 

            # update historical power measurement for next prediction
            # reverst the historical data to enable FILO: [t, t-1, ...,t-l]
            P_his = self.FILO(P_his,P_pred)

            # Save step-wise power
            P_pred_ph.append(P_pred) # save all step-wise power for cost calculation

            ### =================================================================
            ###       Zone Temperatre Predicction
            ### ============================================================
            Ts = 273.15+13 # used as constant in this model
            Tz_pred = self.zone_temperature(alpha_zone, beta_zone, gamma_zone, l, Tz_his, mz, Ts, Toa)

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

        ener_cost = float(np.sum(np.array(price_ph)*np.array(P_pred_ph)))

        #print u_ph,P_pred_ph,penalty

        # objective for a minimization problem
        f = ener_cost + penalty

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
        # call optimizer
        #x0=[0.1]*self.PH
        #lb=[0.1]*self.PH
        #ub=[1.0]*self.PH
        #x_opt,f_opt,nbr_iters,nbr_fevals,solve_time = \
        #    dfo.fmin(obj, xstart=x0,lb=lb,ub=ub,alg=3,nbr_cores=3,x_tol=1e-3,f_tol=1e-6)
        xtol=1e-6
        ftol=1e-6
        #solver='de' # GLP
        solver='ralg'
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

        # bounds
        lb = [0.1]*self.PH
        ub = [1.0]*self.PH

        return GLP(objective, lb=lb, ub=ub, maxIter = 1e5)

    def openopt_model_nlp(self):
        """This is to formulate a nolinear programming problem in openopt: minimization
        """
        objective = lambda u: self.obj([u[i] for i in range(self.PH)])
        # objective gradient - optional
        df = None
        # start point
        start = 0.5*np.ones(self.PH)
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
        lb = [0.1]*self.PH
        ub = [1.0]*self.PH

        # tolerance control
        # required constraints tolerance, default for NLP is 1e-6
        contol = 1e-6

        # If you use solver algencan, NB! - it ignores xtol and ftol; using maxTime, maxCPUTime, maxIter, maxFunEvals, fEnough is recommended.
        # Note that in algencan gtol means norm of projected gradient of  the Augmented Lagrangian
        # so it should be something like 1e-3...1e-5
        gtol = 1e-7 # (default gtol = 1e-6)

        # see https://github.com/troyshu/openopt/blob/d15e81999ef09d3c9c8cb2fbb83388e9d3f97a98/openopt/oo.py#L390.
        return NLP(objective, start, df=df,  c=c,  dc=dc, h=h,  dh=dh,  A=A,  b=b,  Aeq=Aeq,  beq=beq,  
        lb=lb, ub=ub, gtol=gtol, contol=contol, maxIter = 10000, maxFunEvals = 1e7, name = 'NLP for: '+str(self.time))

    def get_optimization_model(self):
        #return self.openopt_model_glp()
        return self.openopt_model_nlp()

    def FILO(self,lis,x):
        lis.pop() # remove the last element
        lis.reverse()
        lis.append(x)
        lis.reverse()

        return lis

    def zone_temperature(self,alpha, beta, gamma, l, Tz_his, mz, Ts, Toa):
        """Predicte zone temperature at next step

        :param alpha: coefficients from curve-fitting
        :type alpha: list
        :param beta: coefficient from curve-fitting
        :type beta: scalor
        :param gamma: coefficient from curve-fitting
        :type gamma: scalor
        :param l: historical step
        :type l: scalor
        :param Tz_his: historical zone temperature array
        :type Tz_his: list
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
        if int(l) != len(alpha) or int(l) != len(Tz_his):
            raise ValueError("'l' is not equal to the size of historical zone temperature or the coefficients.")
        # conver to numpy array
        Tz_his = np.array(Tz_his).reshape(1,-1)
        alpha = np.array(alpha).reshape(-1)
        Tz = (np.sum(alpha*Tz_his,axis=1) + beta*mz*Ts + gamma*Toa)/(1+beta*mz)

        return float(Tz)

    def total_power(self,alpha, beta, gamma, l, P_his, mz, Toa):
        """Predicte zone temperature at next step

        :param alpha: coefficients from curve-fitting
        :type alpha: list
        :param beta: coefficient from curve-fitting
        :type beta: list
        :param gamma: coefficient from curve-fitting
        :type gamma: list
        :param l: historical step
        :type l: scalor
        :param P_his: historical zone temperature array
        :type P_his: list
        :param mz: zone air mass flowrate at time t
        :type mz: scalor
        :param Toa: outdoor air dry bulb temperature at time t
        :type Toa: scalor

        :return: predicted system power at time t
        :rtype: scalor
        """
        # check dimensions
        if int(l) != len(alpha) or int(l) != len(P_his):
            raise ValueError("'l' is not equal to the size of historical zone temperature or the coefficients.")
        # conver to numpy array
        P_his = np.array(P_his).reshape(1,-1)
        alpha = np.array(alpha).reshape(-1)   
        beta = np.array(beta).reshape(-1) 
        gamma = np.array(gamma).reshape(-1)      
        #perform prediction     
        P = (np.sum(alpha*P_his,axis=1) + beta[0]*mz+beta[1]*mz**2 + gamma[0]+ gamma[1]*Toa+gamma[2]*Toa**2)
        # need improve this part when power is negative, cannot assign it to 0 because for optimization problem this would lead to minimum cost 0
        # cannot assign a big number either because the historical value will be used for next step prediction
        P = abs(P) 
        return float(P)

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