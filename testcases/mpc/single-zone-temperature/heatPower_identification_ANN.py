from __future__ import division
import time
import numpy as np
import sklearn
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
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
var_name = 'T_roo'
l = 4
var_his = pd.DataFrame(data[var_name])
for i in range(l):
    var_his[var_name+'_'+str(i+1)] = data[var_name].values
    shift = var_his[var_name+'_'+str(i+1)].shift(periods=i+1)
    var_his[var_name+'_'+str(i+1)]=shift.values
var_his=var_his.drop(columns=[var_name])

# remove NANs
data=pd.concat([data,var_his],axis=1)
data.dropna(inplace=True)
#data.to_csv('prepared_data_power.csv')

print data
X= data[[var_name+'_1',var_name+'_2',var_name+'_3',var_name+'_4','T_roo','T_set','T_oa']].values
y= data['P_hea'].values

# split traing and testing data
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)

print X_train
print y_train

# #############################################################################
# Fit regression model
# First, normalize the data. Create a scaler based on training data
scaler = StandardScaler().fit(X_train)

# Dimension reduction

# Second, create a ANN estimator
ann = MLPRegressor(solver='lbfgs',alpha=0.001)

# Third, create steps
steps = [('normalize',scaler),('reg',ann)]

# Foruth, create pipeline for evaluation
pipe = Pipeline(steps)

# Fifth, perform model selection using grid search to find the best trained ANN model
estimator = GridSearchCV(pipe,
                   param_grid={'reg__alpha':[1e-4,1e-3,1e-2,1e-1,1],
                               'reg__activation':['identity','logistic','tanh','relu']},
                   cv=5,scoring='neg_mean_squared_error')

# fit the model using grid searching validation
t0 = time.time()
estimator.fit(X_train, y_train)
ann_fit = time.time() - t0
print("ANN complexity and bandwidth selected and model fitted in %.3f s"
      % ann_fit)

t0 = time.time()
y_ann = estimator.predict(X_test)
ann_predict = time.time() - t0
print("ANN prediction for %d inputs in %.3f s"
      % (X_test.shape[0], ann_predict))
# #############################################################################
# train  


# Look at the accuracy
r2 = r2_score(y_test,y_ann)
mse = mean_squared_error(y_test,y_ann)
def nrmse(y,ypred):
      mse = mean_squared_error(y,ypred)
      return np.sqrt(np.sum(mse))/np.mean(y+1e-06)

nr_mse = nrmse(y_test, y_ann)
accuracy = {'r2':r2,'mse':mse, 'nrmse':nr_mse}

with open('powerHeaANN.json', 'w') as json_file:
      json.dump(accuracy,json_file)

plt.scatter(y_test, y_ann, c='b',label='Power')
plt.xlabel('Measurement [W]')
plt.ylabel('Prediction [W]')
plt.xlim([0,10000])
plt.ylim([0,10000])
plt.legend()
plt.savefig('power-heat-scatter.pdf')
# Visualize training and prediction 
plt.figure()
plt.subplot(211)
plt.plot(y_test,'b-',lw=0.5,label='Target')
plt.plot(y_ann,'r--',lw=0.5,markevery=0.05,marker='o',markersize=2,label='ANN')
plt.ylabel('Power [W]')
plt.legend()

plt.subplot(212)
plt.plot(y_ann-y_test, 'r-', lw=0.5, label='ANN')
plt.ylabel('Error (W)')
plt.legend()

plt.savefig('power-heat.pdf')


# save the model
joblib.dump(estimator,'powerHeaANN.pkl')


