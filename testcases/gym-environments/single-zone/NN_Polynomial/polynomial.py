# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 10:18:49 2023

@author: Mingyue Guo
"""

#%% import
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import json
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
#%% load data
def load_data():
    data = pd.read_csv('train_data.csv',index_col=[0])
    return data
    
def data_split(data):
    x = data[['mass_flow', 'T_roo', 'T_sa']].values
    y = data['P_tot'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    return x_train, x_test, y_train, y_test


def polynomial(alpha, m, Tz, Ts):
    alpha=np.array(alpha).reshape(-1)
    #beta=np.array(beta).reshape(-1)
    P = alpha[0]+alpha[1]*m+alpha[2]*m**2+alpha[3]*m**3 +(1008./3)*m*(Tz-Ts)#+ beta[0]+ beta[1]*Toa+beta[2]*Toa**2

    return P

def func_P(x,alpha1,alpha2,alpha3,alpha4):
    alpha = np.array([alpha1,alpha2,alpha3,alpha4])
    x1 = x[:,0]
    x2 = x[:,1]-273.15
    x3 = 14#x[:,2]-273.15
    y = polynomial(alpha, x1, x2, x3)
    return y

def plot_scatter(ypred_test, x_test, y_test):
    plt.scatter(x_test[:,0], y_test-((1008./3)*x_test[:,0]*(x_test[:,1]-x_test[:,2])))
    plt.scatter(x_test[:,0], ypred_test)

def identify_paras(x_train, x_test, y_train, y_test):
    # bounds=([0,0,0, 0],[np.inf]*4)
    # popt,pcov = curve_fit(func_P,x_train,y_train, bounds=bounds)
    popt,pcov = curve_fit(func_P,x_train,y_train, bounds=(0, np.inf))
    ypred_train = func_P(x_train,*popt)

    # test on testing data
    ypred_test = func_P(x_test,*popt)

    ### plot fitting results
    fig = plt.figure(figsize = (6,8))
    plt.subplot(311)
    plt.plot(y_train,'b-',lw=0.5,label='Target in Testing')
    plt.plot(ypred_train,'r--',lw=0.5,markevery=0.05,marker='o',markersize=2,label='Prediction in Training')
    plt.ylabel('Power (W)')
    plt.legend(bbox_to_anchor=(1.05,1.0))
    
    plt.subplot(312)
    plt.plot(y_test,'b-',lw=0.5,label='Target in Testing')
    plt.plot(ypred_test,'r--',lw=0.5,markevery=0.05,marker='o',markersize=2,label='Prediction in Training')
    plt.ylabel('Power (W)')
    plt.legend(bbox_to_anchor=(1.05,1.0))
    
    plt.subplot(313)
    plt.plot(ypred_test-y_test, 'b-', label='Prediction Errors in Testing')
    plt.ylabel('Absolute Error (W)')
    plt.xlabel('Index')
    plt.legend(bbox_to_anchor=(1.05,1.0))
    fig.savefig(r'polynomial_result.png', dpi =400, bbox_inches = 'tight')

    # accurancy of 7 days
    fig = plt.figure(figsize = (6,8))
    plt.subplot(311)
    plt.plot(y_train[:288],'b-',lw=0.5,label='Target in Testing')
    plt.plot(ypred_train[:288],'r--',lw=0.5,markevery=0.05,marker='o',markersize=2,label='Prediction in Training')
    plt.ylabel('Power (W)')
    plt.legend(bbox_to_anchor=(1.05,1.0))
    
    plt.subplot(312)
    plt.plot(y_test[:288],'b-',lw=0.5,label='Target in Testing')
    plt.plot(ypred_test[:288],'r--',lw=0.5,markevery=0.05,marker='o',markersize=2,label='Prediction in Training')
    plt.ylabel('Power (W)')
    plt.legend(bbox_to_anchor=(1.05,1.0))
    
    plt.subplot(313)
    plt.plot(ypred_test[:288]-y_test[:288], 'b-', label='Prediction Errors in Testing')
    plt.ylabel('Absolute Error (W)')
    plt.xlabel('Index')
    plt.legend(bbox_to_anchor=(1.05,1.0))
    fig.savefig(r'polynomial_result_1day.png', dpi =400, bbox_inches = 'tight')


    ### export model parameter
    popt_zone = {'alpha':list(popt)}
    
    with open('power.json', 'w') as fp:
        json.dump(popt_zone, fp)
    
    # Look at the accuracy
    r2 = r2_score(y_test,ypred_test)
    mse = mean_squared_error(y_test,ypred_test)
    def nrmse(y,ypred):
          mse = mean_squared_error(y,ypred)
          return np.sqrt(np.sum(mse))/np.mean(y+1e-06)
    
    nr_mse = nrmse(y_test, ypred_test)
    accuracy = {'r2':r2,'mse':mse, 'nrmse':nr_mse}

    return popt,pcov,accuracy

def plot_draft(data):
    cos = ['speed', 'T_roo', 'P_tot']
    for c in cos:
        if 'T_' in c:
            (data[c]-273.15).plot()
            plt.ylabel(c+' [C]')
        else:
            data[c].plot()
            plt.ylabel(c)
        plt.legend()
        plt.xlabel('TimeIndex')
        plt.show()
    
#%% main
if __name__ == '__main__':
    data = load_data()
    x_train, x_test, y_train, y_test = data_split(data)
    popt,pcov,accuracy = identify_paras(x_train, x_test, y_train, y_test)
