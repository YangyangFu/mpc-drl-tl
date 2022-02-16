from __future__ import print_function
from __future__ import absolute_import, division

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import json
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def total_power(alpha, mz):
    """Predicte power at next step

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

    alpha=np.array(alpha).reshape(-1)
    #beta=np.array(beta).reshape(-1)
    P = alpha[0]+alpha[1]*mz+alpha[2]*mz**2+alpha[3]*mz**3 #+ beta[0]+ beta[1]*Toa+beta[2]*Toa**2

    return P

data = pd.read_csv('train_data.csv',index_col=[0])
x = data['mass_flow'].values
y = data['P_tot'].values

### prepare data for fitting
# split data randomly
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# call curve fit
# fit a power model
def func_P(x,alpha1,alpha2,alpha3,alpha4):

    alpha = np.array([alpha1,alpha2,alpha3,alpha4])
    y = total_power(alpha,x)

    return y

popt,pcov = curve_fit(func_P,x_train,y_train, bounds=(0,np.inf))
ypred_train = func_P(x_train,*popt)

# test on testing data
ypred_test = func_P(x_test,*popt)

### plot fitting results
plt.figure()
plt.subplot(311)
plt.plot(y_train,'b-',lw=0.5,label='Target in Testing')
plt.plot(ypred_train,'r--',lw=0.5,markevery=0.05,marker='o',markersize=2,label='Prediction in Training')
plt.ylabel('Power (W)')
plt.legend()

plt.subplot(312)
plt.plot(y_test,'b-',lw=0.5,label='Target in Testing')
plt.plot(ypred_test,'r--',lw=0.5,markevery=0.05,marker='o',markersize=2,label='Prediction in Training')
plt.ylabel('Power (W)')
plt.legend()

plt.subplot(313)
plt.plot(ypred_test-y_test, 'b-', label='Prediction Errors in Testing')
plt.ylabel('Error (W)')
plt.legend()

plt.savefig('power.pdf')


### check robustness
x_fit = np.arange(0,0.4,0.01)
print(x_fit)
y_fit = [func_P(i,*popt) for i in x_fit]

fig=plt.figure(figsize=[6,6])
plt.scatter(x,y)
plt.plot(x_fit,y_fit,'k-')
plt.xlabel('mass flowrate [kg/s]')
plt.ylabel('power [W]')
plt.savefig('fan-power-fit.png')
plt.savefig('fan-power-fit.pdf')

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

with open('power_accuracy.json', 'w') as json_file:
      json.dump(accuracy,json_file)