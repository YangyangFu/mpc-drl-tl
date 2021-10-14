from __future__ import print_function
from __future__ import absolute_import, division

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import json
from model import Zone

data = pd.read_csv('Tz_core_data.csv',index_col=[0])

# split training and testing
ratio = 0.8
n_train = int(ratio*len(data))
data_train = data.iloc[:n_train,:]
data_test = data.iloc[n_train:,:]

# model settings
lz = 4
lo = 4

# fit a zone temperature model
zone = Zone(Lz=lz, Lo=lo)

def predict(zone,Tz_mea_his, To_mea_his, mz, Ts, params):
    N = len(mz)
    i = max(zone.Lz, zone.Lo)
    zone.params = params
    Tz_t_his_pred = Tz_mea_his[i,:]
    Tz_pred = list(Tz_t_his_pred)
    while i < N:
        Tz_t_his = Tz_mea_his[i,:]
        mz_t = mz[i]
        To_t_his = To_mea_his[i,:]
        Ts_t = Ts[i]

        Tz_t_pred = zone.predict(Tz_t_his, Tz_t_his_pred, To_t_his, mz_t, Ts_t)

        # update historical Tz prediction for next step
        Tz_t_his_pred =np.append(Tz_t_his_pred,Tz_t_pred)[1:]

        # update prediction overall
        Tz_pred.append(Tz_t_pred)

        # update step
        i += 1
    return np.array(Tz_pred)

def func_TZone(x,alpha1,alpha2,alpha3,alpha4,beta1,beta2,beta3,beta4,gamma):
    alpha= np.array([alpha1,alpha2,alpha3,alpha4])
    beta= np.array([beta1,beta2,beta3,beta4])
    Tz_his = x[:,:lz]
    To_his = x[:, lz:lz+lo]
    mz = x[:,lz+lo]
    Ts = x[:,lz+lo+1]
    params = {'alpha':alpha, 'beta':beta, 'gamma':gamma}
    y = predict(zone,Tz_his, To_his, mz, Ts, params)
    print (y)
    print (y.shape)
    return y

# represent data in np.array
x_train = data_train[['Tz_4','Tz_3','Tz_2','Tz_1', 'To_4','To_3','To_2','To_1', 'mz', 'Tsa']].values
y_train = data_train['Tz'].values

popt,pcov = curve_fit(func_TZone,x_train,y_train)
ypred_train = func_TZone(x_train,*popt)

# test on testing data
x_test = data_test[['Tz_4','Tz_3','Tz_2','Tz_1', 'To_4','To_3','To_2','To_1', 'mz', 'Tsa']].values
y_test = data_test['Tz'].values
ypred_test = func_TZone(x_test,*popt)

plt.figure()
plt.subplot(311)
plt.plot(y_train,'b-',lw=0.5,label='Target in Training')
plt.plot(ypred_train,'r--',lw=0.5,markevery=0.05,marker='o',markersize=2,label='Prediction in Training')
plt.ylabel('Temperature (C)')
plt.legend()

plt.subplot(312)
plt.plot(y_test,'b-',lw=0.5,label='Target in Testing')
plt.plot(ypred_test,'r--',lw=0.5,markevery=0.05,marker='o',markersize=2,label='Prediction in Testing')
plt.ylabel('Temperature (C)')
plt.legend()

plt.subplot(313)
plt.plot(ypred_test-y_test, 'b-', label='Prediction Errors in Testing')
plt.ylabel('Error (C)')
plt.legend()

plt.savefig('TZone.pdf')


# export model parameter
popt_zone = {'alpha':list(popt[:lz]),
            'beta':list(popt[lz:lz+lo]),
            'gamma':popt[lz+lo]}

with open('zone.json', 'w') as fp:
    json.dump(popt_zone, fp)