import numpy as np
import json

class Zone():
    """Autoregression exogenuous models for zone temperature prediction
    """
    def __init__(self, Lz, Lo, json_file=""):

        self.Lz = Lz # zone temperature delays
        self.Lo = Lo # outdoor air temperature delays
        self.json_file = json_file # result file
        if json_file:
            fjson = open(json_file)
            self.params = json.load(fjson)
        else:
            self.params = {'alpha':[], 'beta':[], 'gamma':[]}

    def model(self,Tz_his_meas, To_his_meas, mz, Tsa):
        """Zone temperature ARX model without auto-recorrection term as defined below:

        $$T_z^{t+1} = \sum_{j=0}^{lz-1} \alpha_jT_z^{t-j} + \sum_{j=0}^{lo-1} \beta_jT_o^{t-j} + \gamma\dot m_z^{t=1}(T_s^{t=1}-T_z^{t+1})$$

        :param Tz_his_meas: Historical zone temperature at time step t+1. The last element represents the latest measurement, e.g., [T(t-(L-1)),...,T(t-1),T(t)].
        :type Tz_his_meas: list or np.array in (L,)
        :param To_his_meas: Historical outdoor air temperature at time step t+1. The last element represents the latest measurement, e.g., [T(t-(L-1)),...,T(t-1),T(t)].
        :type To_his_meas: list or np.array in (L,)
        :param mz: mass flow rate of zonal supply air at time step t+1 - [kg/s]
        :type mz: scalor 
        :param Tsa: supply air temperature at time step t+1 - [C]
        :type Tsa: scalor
    
        :return:
            predicted temperature for time step t+1
            
        """
        # check key parameters
        params = self.params
        if 'alpha' not in params or 'beta' not in params or 'gamma' not in params:
            raise ValueError("zone temperature ARX model parameters 'alpha', 'beta' or 'gamma' are not specified by users!!")
        else:
            alpha = params['alpha'] # list
            beta = params['beta'] # list
            gamma = params['gamma'] # list

        # check data dimensions
        if int(self.Lz) != len(alpha) or int(self.Lz) != len(np.array(Tz_his_meas)):
            raise ValueError("The delayed step 'l' is not equal to the size of historical zone temperature or the length of coefficient 'alpha'.")
        if int(self.Lo) != len(beta) or int(self.Lo) != len(np.array(To_his_meas)):
            raise ValueError("The delayed step 'l' is not equal to the size of historical zone temperature or the length of coefficient 'alpha'.")
        # check data type ?/
        Tz_pred = np.sum(np.array(alpha)*np.array(Tz_his_meas)) + np.sum(np.array(beta)*np.array(To_his_meas)) + gamma*mz*(Tsa-Tz_his_meas[-1])

        return float(Tz_pred)
            
    def autocorrection(self,Tz_his_meas, Tz_his_pred):
        """This is to add auto-correction term to the predicted zone air temperature
         
         $$\dot q_z^{t+1} = \sum_{j=0}^{l-1} \frac{T_z^{t-j}-\hat T_z^{t-j}}{l}$$

        :param Tz_his_meas: Historical zone temperature measurement at time step t+1. The last element represents the latest measurement, e.g., [T(t-(Lz-1)),...,T(t-1),T(t)]
        :type Tz_his_meas: list or np.array in (L,)
        :param Tz_his_pred: historical zone temperature prediction at time stept t+1. The last element represents the latest prediction, e.g., [T(t-(Lz-1)),...,T(t-1),T(t)]
        :type Tz_his_pred: list or np.array in (L,)
        """

        #Tz_pred = self.model(Tz_mea, mz, Toa, l, params)+ np.sum(np.array(Tz_mea)-np.array(Tz_pred),axis=1)/l
        correction = np.mean(np.array(Tz_his_meas)-np.array(Tz_his_pred))

        return float(correction)     

        
    def predict(self,Tz_his_meas, Tz_his_pred, To_his_meas, mz, Tsa):
        """This is to provide a complete prediction with autocorrection term

        $$T_z^{t+1} = T_z^{t+1} + \dot q_z^{t+1}$$ 

        :param Tz_his_meas: Historical zone temperature measurement at time step t+1, [Tz_mea(t-(Lz-1)), ..., Tz_mea(t-2), Tz_mea(t)] 
        :type Tz_his_meas: list or np.array in (Lz,)
        :param Tz_his_pred: historical zone temperature prediction at time stept t+1, [Tz_pred(t-(Lz-1)), ..., Tz_pred(t-2), Tz_pred(t)] 
        :type Tz_his_pred: list or np.array in (Lz,)
        :param To_his_meas: Historical outdoor air temperature at time step t+1. The last element represents the latest measurement, e.g., [T(t-(Lo-1)),...,T(t-1),T(t)].
        :type To_his_meas: list or np.array in (Lo,)
        :param mz: mass flow rate of zonal supply air at time step t - [kg/s]
        :type mz: scalor 
        :param Tsa: supply air temperature at time step t - [C]
        :type Tsa: scalor
    
        :return:
            predicted temperature for time step t+1
        """
        Tz_pred_next = self.model(Tz_his_meas, To_his_meas, mz, Tsa)
        correction =self.autocorrection(Tz_his_meas, Tz_his_pred)

        Tz_pred_next += correction

        return float(Tz_pred_next)

class FanPower():
    """Polynominal model for AHU fan power prediction
    """
    def __init__(self, n=4, json_file=""):
        """ initialize model
        :param n: number of the polynomial equation. default to 2
        :type n: int, optional
        :param json_file: file path of the parameters in json file. Used for MPC function calling. defaults to " "
        :type json_file: str, optional
        """
        self.n = n
        if json_file:
            fjson = open(json_file)
            self.params = json.load(fjson)
        else:
            self.params = {'alpha':[]}
    
    def model(self, mz):
        """ Equation of fan power model
                p=\sum_{i=0}^{n}\alpha[n]*mz^n
        :param mz: [description]
        :type mz: [type]
        """
        # check key parameters
        params = self.params
        if 'alpha' not in params :
            raise ValueError("fan power polynomial parameters 'alpha' are not specified by users!!")
        else:
            alpha = params['alpha'] # list

        # check data dimensions
        if int(self.n) != len(alpha):
            raise ValueError("The specified polynomial equation order is not equal to the length of coefficient 'alpha'.")

        # check data type ?/

        # calculate power
        p = 0
        for i in range(self.n):
            p += alpha[i]*mz**i
        
        return p 

    def predict(self, mz):
        """predict the fan power based on inputs

        :param mz: mass flow rate of fan 
        :type mz: float
        :return: fan power
        :rtype: float
        """
        return self.model(mz)

class ChillerPlantPower:
    pass