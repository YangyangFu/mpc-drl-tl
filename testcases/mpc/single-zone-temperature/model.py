import numpy as np
import json
from scipy.integrate import solve_ivp
import pandas as pd 

# %% Define zone temperature RC model
class ZoneTemperature():
    """RC model (4R3C) for zone temperature prediction
        xdot = Ax + Bu
        y = Cx
     States:
        x= [Tai, Twe, Twi]
     Disturbances:
        u = [Tao, qCon_i, qHVAC, qRad_e, qRad_i]
     Output:
        y = [Tai]
    """
    def __init__(self, Rg, Re, Rw, Ri, Cwe, Cwi, Cai, U):
        # assign parameters
        self.Rg = Rg 
        self.Re = Re 
        self.Ri = Ri 
        self.Rw = Rw 
        self.Cwe = Cwe 
        self.Cwi = Cwi 
        self.Cai = Cai
        # initialize the matrix 
        self.A = np.zeros(3,3)
        self.B = np.zeros(3,5)
        self.C = np.zeros(1,3)
        self.D = np.zeros(1,1)
        self.update_matrix()

        # initialize ODE
        # disturbances
        self.U = U # dataframe that stores the disturbances as columns with time as index
        self.xdot = np.zeros(3,1)
        self.solver = "RK45" # options are: RK45, RK23, DOP53, Radau, BDF, LSODA
    def update_matrix(self):
        self.A[0,0] = -1/self.Cai*(1/self.Rg+1/self.Ri)
        self.A[0,2] = 1/(self.Cai*self.Ri)
        self.A[1,1] = -1/self.Cwe*(1/self.Re+1/self.Rw)
        self.A[1,2] = 1/(self.Cwe*self.Rw)
        self.A[2,0] = 1/(self.Cwi*self.Ri)
        self.A[2,1] = 1/(self.Cwi*self.Rw)
        self.A[2,2] = -1/self.Cwi*(1/self.Rw+1/self.Ri)

        self.B[0,0] = 1/(self.Cai*self.Rg)
        self.B[0,1] = 1/self.Cai 
        self.B[0,2] = 1/self.Cai 
        self.B[1,0] = 1/(self.Cwe*self.Re)
        self.B[1,3] = 1/self.Cwe
        self.B[2,4] = 1/self.Cwi

        self.C[0,0] = 1

        self.D = 0

    def u(self,t):
        """return disturbance signal at time t from given source
    
        """
        return self.U.loc[t,:].values()

    def R4C3_ODE(self, t, x):
        x = np.reshape(x, (3,1))        
        ut = np.reshape(self.u(t), (5,1))     
        self.xdot = np.matmul(self.A, x) + np.matmul(self.B, ut)

        # return xdot
        return self.xdot

    def solve(self, t_span, x0, solver, t_eval):
    
        res = solve_ivp(self.R4C3_ODE, 
                t_span = t_span, 
                y0 = x0, 
                method = solver,
                t_eval = t_eval)
        # return xnext
        return res

    def predict(self, t_span, x0, solver, t_eval):

        sol = self.solve(t_span, x0, solver, t_eval)
        x_next = sol.y[0][-1,:]
        return x_next

# %% Define inverse function for zone temperautre RC model to calculate thermal load from setpoints
class ZoneThermalLoad():
    pass
# %% Define system power model
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