import numpy as np
import pandas as pd 
from scipy.optimize import differential_evolution
import json

class Zone():
    """Autoregression exogenuous models for zone temperature prediction
    """
    def __init__(self, L, json_file=""):

        self.L = L
        self.json_file = json_file # 
        if json_file:
            self.params = json.load(json_file)
        else:
            self.params = {'alpha':[], 'beta':[], 'gamma':[]}

    def model(self,Tz_his_meas, mz, Toa, params):
        """Zone temperature ARX model without auto-recorrection term as defined below:

        $$T_z^{t+1} = \sum_{j=0}^{l-1} \alpha_jT_z^{t-j} + \beta\dot m_z^{t=1}(T_s^{t=1}-T_z^{t+1}) + \gamma T_{oa}^{t+1}$$

        :param Tz_his_meas: Historical zone temperature at time step t+1. The last element represents the latest measurement, e.g., [T(t-(L-1)),...,T(t-1),T(t)].
        :type Tz_his_meas: list or np.array in (L,)
        :param mz: mass flow rate of zonal supply air at time step t+1 - [kg/s]
        :type mz: scalor 
        :param Toa: outdoor air temperature at time step t+1 - [C]
        :type Toa: scalor
        :param params: ARX model parameters for each input - need to be identified from training data
        :type params: dict
            e.g. {"alpha": [], "beta":[], "gamma": []}
        
        :return:
            predicted temperature for time step t+1
            
        """
        # check key parameters
        if 'alpha' not in params or 'beta' not in params or 'gamma' not in params:
            raise ValueError("zone temperature ARX model parameters 'alpha', 'beta' or 'gamma' are not specified by users!!")
        else:
            alpha = params['alpha'] # list
            beta = params['beta'] # list
            gamma = params['gamma'] # list

        # check data dimensions
        if int(self.L) != len(alpha) or int(self.L) != len(Tz_his_meas):
            raise ValueError("The delayed step 'l' is not equal to the size of historical zone temperature or the length of coefficient 'alpha'.")

        # check data type ?/
        
        Tz_pred = np.sum(np.array(alpha)*np.array(Tz_his_meas)) + np.array(beta)*mz + gamma*Toa

        return float(Tz_pred)
            
    def autocorrection(self,Tz_his_meas, Tz_his_pred):
        """This is to add auto-correction term to the predicted zone air temperature
         
         $$\dot q_z^{t+1} = \sum_{j=0}^{l-1} \frac{T_z^{t-j}-\hat T_z^{t-j}}{l}$$

        :param Tz_his_meas: Historical zone temperature measurement at time step t+1. The last element represents the latest measurement, e.g., [T(t-(L-1)),...,T(t-1),T(t)]
        :type Tz_his_meas: list or np.array in (L,)
        :param Tz_his_pred: historical zone temperature prediction at time stept t+1. The last element represents the latest prediction, e.g., [T(t-(L-1)),...,T(t-1),T(t)]
        :type Tz_his_pred: list or np.array in (L,)
        """

        #Tz_pred = self.model(Tz_mea, mz, Toa, l, params)+ np.sum(np.array(Tz_mea)-np.array(Tz_pred),axis=1)/l
        correction = np.mean(np.array(Tz_his_meas)-np.array(Tz_his_pred))

        return float(correction)     

        
    def predict(self,Tz_his_meas, Tz_his_pred, mz, Toa, params):
        """This is to provide a complete prediction with autocorrection term

        $$T_z^{t+1} = T_z^{t+1} + \dot q_z^{t+1}$$ 

        :param Tz_his_meas: Historical zone temperature measurement at time step t+1, [Tz_mea(t-(L-1)), ..., Tz_mea(t-2), Tz_mea(t)] 
        :type Tz_his_meas: list or np.array in (L,)
        :param Tz_his_pred: historical zone temperature prediction at time stept t+1, [Tz_pred(t-(L-1)), ..., Tz_pred(t-2), Tz_pred(t-1)] 
        :type Tz_his_pred: list or np.array in (L,)
        :param mz: mass flow rate of zonal supply air at time step t+1 - [kg/s]
        :type mz: scalor 
        :param Toa: outdoor air temperature at time step t+1 - [C]
        :type Toa: scalor
        :param params: ARX model parameters for each input - need to be identified from training data
        :type params: dict
            e.g. {"alpha": [], "beta":[], "gamma": []}
        
        :return:
            predicted temperature for time step t+1
        """
        Tz_pred_next = self.model(Tz_his_meas, mz, Toa, params)
        correction =self.autocorrection(Tz_his_meas, Tz_his_pred)

        Tz_pred_next += correction

        return float(Tz_pred_next)

