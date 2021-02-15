'''

'''
import numpy as np

class mpc_case():
    def __init__(self,PH,CH,time,dt,parameters_zone, parameters_power,measurement,states,predictor):

        self.PH=PH # prediction horizon
        self.CH=CH # control horizon
        self.dt = dt # time step
        self.time = time # current time index

        self.parameters_zone = parameters_zone # parameters for zone dynamic mpc model
        self.parameters_power = parameters_power # parameters for system power dynamic mpc model
        self.measurement = measurement # measurement at current time step
        self.predictor = predictor # price and outdoor air temperature for the future horizons

        self.states = states # dictionary
        self.P_his_t = []
        self.Tz_his_t = []

        self.number_zone = 1
        self.occ_start = 6 # occupancy starts
        self.occ_end = 19 # occupancy ends

    def optimize(self):
        """
        MPC optimizer: call external optimizer to solve a minimization problem
        
        """

    
    def cost(self,u_ph, time):
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
        T_upper[6:19] = 25.0
        T_lower = np.array([18.0 for i in range(24)])
        T_lower[6:19] = 23.0

        overshoot = []
        undershoot = []

        # loop over the prediction horizon
        i = 0
        P_pred_ph = []
        Tz_pred_ph = [] 

        # get current measurement 
        P_his = np.array(self.states['P_his_t']) # current state of historical power [P(t-1) to P(t-l)]   
        Tz_his = np.array(self.states['Tz_his_t']) # current zone temperatur states for zone temperture prediction model  

        # initialize the cost function
        penalty = []  # temperature violation penalty for each zone
        alpha_up = 200.0
        alpha_low = 200.0

        while i < self.PH:

            mz = u_ph[i] # control inputs
            Toa = self.predictor['Toa'][i] # predicted outdoor air temperature

            ### ====================================================================
            ###            Power Predictor
            ### =====================================================================
            # predict total power at current step
            P_pred = self.total_power(alpha_power, beta_power, gamma_power, l, P_his, mz, Toa) 

            # update historical power measurement for next prediction
            # reverst the historical data to enable LIFO
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
                overshoot.append(max((Tz_pred -273.15) - T_upper[t], 0.0))
                undershoot.append(max(T_lower[t] - (Tz_pred-273.15), 0.0))
            
            ### ===========================================================
            ###      Update for next step
            ### ==========================================================
            # update clock
            i+=1

        # calculate total cost based on predicted energy prices
        price_ph = self.predictor['price']

        # zone temperature bounds penalty
        penalty.append(alpha_up *
                       sum(overshoot) + alpha_low * sum(undershoot))

        ener_cost = np.sum(np.array(price_ph)*np.array(P_pred_ph))

        # for a minimization problem
        return ener_cost + sum(penalty)


    def FILO(self,array_1d,x):
        array_list=list(array_1d)
        array_list.pop() # remove the last element
        array_list.reverse()
        array_list.append(x)
        array_list.reverse()
        array=np.array(array_list)

        return array

    def zone_temperature(self,alpha, beta, gamma, l, Tz_his, mz, Ts, Toa):
        """Predicte zone temperature at next step

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
        if int(l) != len(alpha) or int(l) != len(Tz_his):
            raise ValueError("'l' is not equal to the size of historical zone temperature or the coefficients.")

        Tz = (sum(alpha*Tz_his) + beta*mz*Ts + gamma*Toa)/(1+beta*mz)
        return Tz

    def total_power(self,alpha, beta, gamma, l, P_his, mz, Toa):
        """Predicte zone temperature at next step

        :param alpha: coefficients from curve-fitting
        :type alpha: np array (l,)
        :param beta: coefficient from curve-fitting
        :type beta: np array (2,)
        :param gamma: coefficient from curve-fitting
        :type gamma: np array (3,)
        :param l: historical step
        :type l: scalor
        :param P_his: historical zone temperature array
        :type P_his: np array (l,)
        :param mz: zone air mass flowrate at time t
        :type mz: scalor
        :param Toa: outdoor air dry bulb temperature at time t
        :type Toa: scalor

        :return: predicted system power at time t
        :rtype: scalor
        """
        # check dimensions
        if int(l) != len(alpha) or int(l) != P_his.shape[1]:
            raise ValueError("'l' is not equal to the size of historical zone temperature or the coefficients.")

        P = (np.sum(alpha*P_his,axis=1) + beta[0]*mz+beta[1]*mz**2 + gamma[0]+ gamma[1]*Toa+gamma[2]*Toa**2)
        return P
