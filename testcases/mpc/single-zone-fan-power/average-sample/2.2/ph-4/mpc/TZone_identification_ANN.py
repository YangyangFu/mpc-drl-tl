from __future__ import division
import time
import numpy as np
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
# #############################################################################
#read data for the systems
# define the path


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
#X= data[['T_roo_1','T_roo_2','T_roo_3','T_roo_4','T_roo_5','T_roo_6','T_roo_7','T_roo_8','mass_flow','T_oa']].values
X= data[['T_roo_1','T_roo_2','T_roo_3','T_roo_4','mass_flow','T_oa']].values
#X= data[['T_roo_1','mass_flow','T_oa']].values
y= data['T_roo'].values

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
                   cv=5,scoring='neg_mean_absolute_error')

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


# Look at the results
r2_ann = r2_score(y_test,y_ann)
relErr_ann = (y_test-y_ann)/y_test
print r2_ann


plt.scatter(y_test-273.15, y_ann-273.15, c='b',label='Zone Temperature')
plt.xlabel('Temperature [C]')
plt.ylabel('Temperature [C]')
plt.xlim([12,37])
plt.ylim([12,37])
plt.legend()
plt.savefig('TZone-scatter.pdf')
# Visualize training and prediction 
plt.figure()
plt.subplot(211)
plt.plot(y_test-273.15,'b-',lw=0.5,label='Target')
plt.plot(y_ann-273.15,'r--',lw=0.5,markevery=0.05,marker='o',markersize=2,label='ANN')
plt.ylabel('Temperature (C)')
plt.legend()

plt.subplot(212)
plt.plot(y_ann-y_test, 'r-', lw=0.5, label='ANN')
plt.ylabel('Error (C)')
plt.legend()

plt.savefig('TZone.pdf')


# save the model
joblib.dump(estimator,'ann.pkl')




