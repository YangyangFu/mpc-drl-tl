# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

import numpy as np
# ipopt
#from pyomo.environ import *

import casadi as ca
import model
import joblib
import json

class PowerCallback(ca.Callback):
    def __init__(self,name, PH, power_model, opts={}):
        ca.Callback.__init__(self)
        self.PH = PH
        self.power_model = power_model
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

    def power_polynomial(self,mz):
        params = self.power_model
        f_P = model.FanPower(n=len(params['alpha']))
        f_P.params = params
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

        return [ca.DM(P_pred_ph)]

def LIFO(array,x):
    # lAST IN FIRST OUT:
    a = np.append(array,x)

    return list(a[1:])

def objective(PH=4,dt=900., w=[1,0.01],predictor={'price':[1.,1.,1.,1.]}):
    """MPC optimization problem in casadi interface
    """
    with open('power.json') as f:
        power_model = json.load(f)

    # instantiate power function
    P = PowerCallback('P', 
                        PH = PH,
                        power_model = power_model,
                        opts={"enable_fd":True})   
            

    # define casadi variables
    n=2
    u = ca.MX.sym("U",n*PH)

    # define objective function
    power_ph=P(u)

    price_ph = predictor['price']
    fval = []
    for k in range(PH):
        fo = w[0]*power_ph[k]*price_ph[k] + w[1]*u[n*k+1]
        fval.append(fo)
    print(fval,type(fval))
    fval=ca.vertcat(*fval)
    print(fval,type(fval))
    fval_sum = ca.sum1(fval)

    obj=ca.Function('fval',[u],[fval_sum])
    print(fval_sum,type(fval_sum))
    print(obj,type(obj))
    
    # define a function for calculate step-wise objective
    y=obj(u)


    #define a nlp
    prob = {'f':y, 'x': u}
    nlp_optimize = ca.nlpsol('solver', 'ipopt', prob)
    u0 = [0.1, 0.0]*PH
    lbx=[0.0, 0.0]*PH
    ubx=[1., 0.1]*PH

    solution = nlp_optimize(x0=u0, lbx=lbx, ubx=ubx)
        

    return solution['x']

print(objective())

