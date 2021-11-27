from __future__ import division
import time
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import json
# #############################################################################
#read data for the systems
# define the path

data = pd.read_csv('train_data.csv',index_col=[0])

# prepare data for zone temperature prediction
def prepare_zone_data(data, lz, lo):
      zone_temp_name = "T_roo"
      Tz_his = pd.DataFrame(data[zone_temp_name]-273.15)
      for i in range(lz):
            Tz_his['Tz_'+str(i+1)] = Tz_his[zone_temp_name].values
            shift = Tz_his['Tz_'+str(i+1)].shift(periods=i+1)
            Tz_his['Tz_'+str(i+1)]=shift.values
      Tz_his['time'] = data.index
      Tz_his['Tz'] = data[zone_temp_name]-273.15
      Tz_his=Tz_his.drop(columns=[zone_temp_name])

      oa_name = "T_oa"
      To_his = pd.DataFrame(data[oa_name]-273.15)
      for i in range(lo):
            To_his['To_'+str(i+1)] = To_his[oa_name].values
            shift = To_his['To_'+str(i+1)].shift(periods=i+1)
            To_his['To_'+str(i+1)]=shift.values
      To_his['To'] = data[oa_name]-273.15
      To_his=To_his.drop(columns=[oa_name])
      # combine them and return
      return pd.concat([Tz_his,To_his],axis=1)

# proprocess data
lz=4
lo=4
# zone
Tz_data = prepare_zone_data(data,lz,lo)
Tz_data['T_set'] = data['T_set'] - 273.15
Tz_data.dropna(inplace=True)
data=Tz_data

# export for future investigation
data.to_csv('prepared_data_tzone.csv')

print(data)
#X= data[['T_roo_1','T_roo_2','T_roo_3','T_roo_4','T_roo_5','T_roo_6','T_roo_7','T_roo_8','mass_flow','T_oa']].values
X= data[['Tz_1','Tz_2','Tz_3','Tz_4','T_set','To']].values
#X= data[['T_roo_1','mass_flow','T_oa']].values
y= data['Tz'].values

# split traing and testing data
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)

# #############################################################################
# Fit regression model
# First, normalize the data. Create a scaler based on training data
scaler = StandardScaler().fit(X_train)

# Dimension reduction
ann = MLPRegressor(solver='adam',alpha=0.01,max_iter=10000)

# Third, create steps
steps = [('normalize',scaler),('reg',ann)]

# Foruth, create pipeline for evaluation
pipe = Pipeline(steps)

# Fifth, perform model selection using grid search to find the best trained ANN model
estimator = GridSearchCV(pipe,
                   param_grid={'reg__alpha':[1e-4,1e-3,1e-2],
                                'reg__hidden_layer_sizes':[(128),(128,128)],
                               'reg__activation':['logistic','relu','tanh']},
                   cv=5,scoring='neg_mean_squared_error')

# fit the model using grid searching validation

estimator.fit(X_train, y_train)

ypred_train = estimator.predict(X_train)
ypred_test = estimator.predict(X_test)

plt.figure()
plt.subplot(211)
plt.plot(y_test,'b-',lw=0.5,label='Target')
plt.plot(ypred_test,'r--',lw=0.5,markevery=0.05,marker='o',markersize=2,label='ANN')
plt.ylabel('Temperature (C)')
plt.legend()
plt.subplot(212)
plt.plot(ypred_test-y_test, 'r-', lw=0.5, label='ANN')
plt.ylabel('Error (C)')
plt.legend()
plt.savefig('TZone_ANN.pdf')

# save the model
joblib.dump(estimator,'zone_ann.pkl')

##  prediction performance
def nrmse(y,ypred):
      mse = mean_squared_error(y,ypred)
      return np.sqrt(np.sum(mse))/np.mean(y+1e-06)

r2_train = r2_score(y_train,ypred_train)
mse_train = mean_squared_error(y_train,ypred_train)
error_mean_train = np.mean(ypred_train-y_train)
nrmse_train = nrmse(y_train, ypred_train)

r2_test = r2_score(y_test,ypred_test)
mse_test = mean_squared_error(y_test,ypred_test)
error_mean_test = np.mean(ypred_test-y_test)
nrmse_test = nrmse(y_test, ypred_test)

accuracy = {'test':{'r2':r2_test,'mse':mse_test, 'nrmse':nrmse_test, 'error_mean':error_mean_test}, 
            'train':{'r2':r2_train,'mse':mse_train, 'nrmse':nrmse_train, 'error_mean':error_mean_train}}

with open('zone_ann_accuracy.json', 'w') as json_file:
      json.dump(accuracy,json_file)