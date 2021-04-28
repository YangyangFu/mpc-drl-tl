import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import json
from coarse_models import Zone

data = pd.read_csv('zone_data.csv',index_col=[0])

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
#data.to_csv('prepared_data_tzone.csv')

print data
# split training and testing
ratio = 0.8
n_train = int(ratio*len(data))
data_train = data.iloc[:n_train,:]
data_test = data.iloc[n_train:,:]

# fit a zone temperature model
zone = Zone(L=l)

def predict(zone,Tz_mea_his,mz, Toa,params):
    N = len(mz)
    i = zone.L
    zone.params = params
    Tz_t_his_pred = Tz_mea_his[i,:]
    Tz_pred = list(Tz_t_his_pred)
    while i < N:
        Tz_t_his = Tz_mea_his[i,:]
        mz_t = mz[i]
        Toa_t = Toa[i]
        Tz_t_pred = zone.predict(Tz_t_his, Tz_t_his_pred, mz_t, Toa_t)

        # update historical Tz prediction for next step
        Tz_t_his_pred =np.append(Tz_t_his_pred,Tz_t_pred)[1:]

        # update prediction overall
        Tz_pred.append(Tz_t_pred)

        # update step
        i += 1
    return np.array(Tz_pred)

def func_TZone(x,alpha1,alpha2,alpha3,alpha4,beta,gamma):
    alpha= np.array([alpha1,alpha2,alpha3,alpha4])
    Tz_his = x[:,:l]
    mz = x[:,l]
    Toa = x[:,l+1]
    params = {'alpha':alpha, 'beta':beta, 'gamma':gamma}
    y = predict(zone,Tz_his, mz, Toa, params)
    print y
    print y.shape
    return y

# represent data in np.array
x_train = data_train[['T_roo_4','T_roo_3','T_roo_2','T_roo_1','mass_flow','T_oa']].values
y_train = data_train['T_roo'].values

popt,pcov = curve_fit(func_TZone,x_train,y_train)
ypred_train = func_TZone(x_train,*popt)

# test on testing data
x_test = data_test[['T_roo_4','T_roo_3','T_roo_2','T_roo_1','mass_flow','T_oa']].values
y_test = data_test['T_roo'].values
ypred_test = func_TZone(x_test,*popt)

plt.figure()
plt.subplot(311)
plt.plot(y_train-273.15,'b-',lw=0.5,label='Target in Training')
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
popt_zone = {'alpha':list(popt[:l]),
            'beta':popt[l],
            'gamma':popt[l+1]}

with open('zone.json', 'w') as fp:
    json.dump(popt_zone, fp)