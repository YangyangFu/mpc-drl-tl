'''

'''
import numpy as np
# optimization package
import pyOpt 
# import ipopt

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
        self.optimum={}

    def optimize(self):
        """
        MPC optimizer: call external optimizer to solve a minimization problem
        
        """
        opt_prob = pyOpt.Optimization('MPC for time'+str(int(self.time)),self.obj)
        opt_prob.addObj('f')

        # Assign design variables
        lb = [0.1]*self.PH
        up = [1.]*self.PH
        value = [0.1]*self.PH

        opt_prob.addVarGroup('u', self.PH, type='c', value=value, lower=lb, upper=up)
        print opt_prob
        # Assign constraints - unconstrained

        # Solve the optimization problem
        # initialize
        self.optimum={}
        
        # call optimizer
        #psqp = pyOpt.PSQP()
        #psqp.setOption('IPRINT',2)
        #[fstr,xstr,info]=psqp(opt_prob,sens_type='FD')
        #nsga2 = pyOpt.NSGA2()
        #nsga2.setOption('PrintOut',0)
        #[fstr,xstr,info]=nsga2(opt_prob)
        solvopt = pyOpt.SOLVOPT()
        solvopt.setOption('iprint',2)
        [fstr,xstr,info]=solvopt(opt_prob,sens_type='FD')

        print fstr, xstr, info
        print opt_prob.solution(0)
        #obj_opt = solution._objectives[0].optimum
        #for var in solution._variables.keys():
        #    u_ph_opt.append(solution._variables[var].value)

        self.optimum['objective'] = fstr
        self.optimum['variable'] = xstr
        
        return self.optimum
        
    def obj(self,u_ph):
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
                overshoot.append(max(float((Tz_pred -273.15) - T_upper[t]), 0.0))
                undershoot.append(max(float(T_lower[t] - (Tz_pred-273.15)), 0.0))
            
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
        g = []

        # simulation status
        fail = 0
        #print f, g
        return f, g, fail

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