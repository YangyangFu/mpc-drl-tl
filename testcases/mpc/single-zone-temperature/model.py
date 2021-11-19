from io import RawIOBase
from matplotlib.pyplot import rgrids
import numpy as np
import json

from numpy.lib.polynomial import RankWarning

class ZoneTemperature():
    """RC model (4R3C) for zone temperature prediction
    """
    def __init__(self):
        self.Rg = Rg 
        self.Re = Re 
        self.Ri = Ri 
        self.Rw = Rw 
        self.Cwe = Cwe 
        self.Cwi = Cwi 
        self.Cai = Cai
        self.A = [] 
        self.B = []
        self.C = []
        self.D = []

    def R4C3_ODE(self, x, u):
        
        pass
        # return xdot

    def RungeKunta(self, x, xdot, tolerance):
    
        pass
        # return xnext

    def predict(self, x, u,tolerance):
        xdot = self.R43C_ODE(x,u) 
        xnext = self.RungeKunta(x, xdot, tolerance)

        return xnext

class ZoneThermalLoad():
    pass

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