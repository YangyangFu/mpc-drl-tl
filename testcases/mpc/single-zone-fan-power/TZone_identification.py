import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import json

def zone_temperature(alpha, beta, gamma, l, Tz_his, mz, Ts, Toa):
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
    if int(l) != len(alpha) or int(l) != Tz_his.shape[1]:
        raise ValueError("'l' is not equal to the size of historical zone temperature or the coefficients.")
    alpha = [alpha[0],0., 0., 0.]
    Tz = (np.sum(alpha*Tz_his,axis=1) + beta*mz*Ts + gamma*Toa)/(1+beta*mz)
    return Tz

data = pd.read_csv('train_data.csv',index_col=[0])

# prepare data for zone temperature prediction
l = 4
Tz_his = pd.DataFrame(data['T_roo'])
for i in range(l):
    Tz_his['T_roo_'+str(i+1)] = data['T_roo'].values
    shift = Tz_his['T_roo_'+str(i+1)].shift(periods=i+1)
    Tz_his['T_roo_'+str(i+1)]=shift.values
Tz_his=Tz_his.drop(columns=['T_roo'])

# remove NANs
data=pd.concat([data,Tz_his],axis=1)
data.dropna(inplace=True)
data.to_csv('prepared_data_tzone.csv')

print data
# split training and testing
data_train = data.iloc[:int(20*24*4),:]
print data_train
data_test = data.iloc[20*24*4:-1,:]

# fit a zone temperature model
def func_TZone(x,alpha1,alpha2,alpha3,alpha4,beta,gamma):
    l = 4
    alpha = np.array([alpha1,alpha2,alpha3,alpha4])
    Tz_his = x[:,:l]
    mz = x[:,l]
    Ts = 13+273.15
    Toa = x[:,l+1]
    y = zone_temperature(alpha, beta, gamma, l, Tz_his, mz, Ts, Toa)

    return y

x_train = data_train[['T_roo_1','T_roo_2','T_roo_3','T_roo_4','mass_flow','T_oa']].values
y_train = data_train['T_roo'].values

popt,pcov = curve_fit(func_TZone,x_train,y_train)
ypred_train = func_TZone(x_train,*popt)

# test on testing data
x_test = data_test[['T_roo_1','T_roo_2','T_roo_3','T_roo_4','mass_flow','T_oa']].values
y_test = data_test['T_roo'].values
ypred_test = func_TZone(x_test,*popt)

plt.figure()
plt.subplot(311)
plt.plot(y_train-273.15,'b-',lw=0.5,label='Target in Testing')
plt.plot(ypred_train-273.15,'r--',lw=0.5,markevery=0.05,marker='o',markersize=2,label='Prediction in Training')
plt.ylabel('Temperature (C)')
plt.legend()

plt.subplot(312)
plt.plot(y_test-273.15,'b-',lw=0.5,label='Target in Testing')
plt.plot(ypred_test-273.15,'r--',lw=0.5,markevery=0.05,marker='o',markersize=2,label='Prediction in Testing')
plt.ylabel('Temperature (C)')
plt.legend()

plt.subplot(313)
plt.plot(ypred_test-y_test, 'b-', label='Prediction Errors in Testing')
plt.ylabel('Error (C)')
plt.legend()

plt.savefig('TZone.pdf')


# export model parameter
popt_zone = {'alpha':list(popt[:4]),
            'beta':popt[4],
            'gamma':popt[5]}

with open('TZone.json', 'w') as fp:
    json.dump(popt_zone, fp)
