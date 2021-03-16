import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import json

def total_power(alpha, beta, gamma, l, P_his, mz, Toa):
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
    if int(l) != len(alpha) or int(l) != P_his.shape[1]:
        raise ValueError("'l' is not equal to the size of historical zone temperature or the coefficients.")
    alpha=[0.0]*l
    #gamma=[0.0]*3
    P = (np.sum(alpha*P_his,axis=1) + beta[0]*mz+beta[1]*mz**2 + gamma[0]+ gamma[1]*Toa+gamma[2]*Toa**2)
    return abs(P)

data = pd.read_csv('train_data.csv',index_col=[0])

# prepare data for zone temperature prediction
l = 4
P_his = pd.DataFrame(data['P_fan'])
for i in range(l):
    P_his['P_fan_'+str(i+1)] = data['P_fan'].values
    shift = P_his['P_fan_'+str(i+1)].shift(periods=i+1)
    P_his['P_fan_'+str(i+1)]=shift.values
P_his=P_his.drop(columns=['P_fan'])

# remove NANs
data=pd.concat([data,P_his],axis=1)
data.dropna(inplace=True)
data.to_csv('prepared_data_power.csv')

print data
# split training and testing
data_train = data.iloc[:int(20*24*4),:]
print data_train
data_test = data.iloc[20*24*4:-1,:]

# fit a zone temperature model
def func_P(x,alpha1,alpha2,alpha3,alpha4,beta1,beta2,gamma0,gamma1,gamma2):
    l = 4
    alpha = np.array([alpha1,alpha2,alpha3,alpha4])
    beta = np.array([beta1,beta2])
    gamma = np.array([gamma0, gamma1, gamma2])
    P_his = x[:,:l]
    mz = x[:,l]
    Toa = x[:,l+1]
    y = total_power(alpha, beta, gamma, l, P_his, mz, Toa)

    return y

x_train = data_train[['P_fan_1','P_fan_2','P_fan_3','P_fan_4','mass_flow','T_oa']].values
y_train = data_train['P_fan'].values

popt,pcov = curve_fit(func_P,x_train,y_train)
ypred_train = func_P(x_train,*popt)

# test on testing data
x_test = data_test[['P_fan_1','P_fan_2','P_fan_3','P_fan_4','mass_flow','T_oa']].values
y_test = data_test['P_fan'].values
ypred_test = func_P(x_test,*popt)

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

plt.savefig('Power.pdf')


# export model parameter
popt_zone = {'alpha':list(popt[:4]),
            'beta':list(popt[4:6]),
            'gamma':list(popt[6:9])}

with open('Power.json', 'w') as fp:
    json.dump(popt_zone, fp)
