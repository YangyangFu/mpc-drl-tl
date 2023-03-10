# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:29:09 2023

@author: Mingyue Guo
"""
#%% import
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse,r2_score as r2

import torchvision.transforms as transforms

from torch.autograd import Variable
#%%
def load_data():
    #path = 'G:\我的云端硬盘\MPCvsDRL\generate_data'
    data = pd.read_csv( 'train_data.csv')
    data.rename(columns = {'Unnamed: 0':'TimeIndex'}, inplace = True)
    return data

def lag_feature(df_pre, window=4, lag_cols=["T_oa"]):
    df = df_pre.copy()
    rolled = df[lag_cols].rolling(window=window, min_periods=0, center=True)
    lag_mean = rolled.mean().reset_index().astype(np.float16)
    lag_max = rolled.quantile(0.95).reset_index().astype(np.float16)
    lag_min = rolled.quantile(0.05).reset_index().astype(np.float16)
    lag_std = rolled.std().reset_index().astype(np.float16)
    for col in lag_cols:
        df[f"{col}_mean_lag{window}"] = lag_mean[col]
        df[f"{col}_max_lag{window}"] = lag_max[col]
        df[f"{col}_min_lag{window}"] = lag_min[col]
        # df[f"{col}_std_lag{window}"] = lag_std[col]
    return df



def feature_engneering(data, his_l, target_co, ph):
    df_pre = data.copy()
    for c in df_pre.columns:
        if 'T_' in c:
            df_pre[c] -= 273.15
    Tz_his = pd.DataFrame(df_pre[['TimeIndex',target_co]])
    for i in range(his_l):
        Tz_his['Tz_'+str(i+1)] = Tz_his[target_co].values
        shift = Tz_his['Tz_'+str(i+1)].shift(periods=i+1)
        Tz_his['Tz_'+str(i+1)]=shift.values
    Tz_his=Tz_his.drop(columns=[target_co])
    df_pre = pd.merge(df_pre, Tz_his, on='TimeIndex')

    # oa_name = 'T_oa'
    # To_his = pd.DataFrame(df_pre[['TimeIndex', oa_name]])
    # for i in range(lo):
    #     To_his['To_'+str(i+1)] = To_his[oa_name].values
    #     shift = To_his['To_'+str(i+1)].shift(periods=i+1)
    #     To_his['To_'+str(i+1)]=shift.values
    # To_his=To_his.drop(columns=[oa_name])
    # df_pre = pd.merge(df_pre, To_his, on='TimeIndex')
    
    df_pre = lag_feature(df_pre, window = 4)
    df_pre.dropna(how = 'any', inplace = True)
    df_pre.drop('TimeIndex', axis = 1, inplace = True)
    
    T_fut = pd.DataFrame(df_pre[target_co])
    for i in range(ph):
        T_fut['Tz_'+str(i+1)] = T_fut[target_co].values
        shift = T_fut['Tz_'+str(i+1)].shift(periods=-(i+1))
        T_fut['Tz_'+str(i+1)]=shift.values
    T_fut.drop(target_co, axis = 1, inplace = True)
    T_fut.dropna(how = 'any', inplace = True)
    
    df_pre = df_pre[:T_fut.shape[0]]

    return df_pre, T_fut


def get_dataset(x_df, y_df, sl, ph, target_co):
    '''
    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    sl : TYPE
        sequence length.
    ph : predict horizon.

    Returns
    -------
    df_3 : TYPE
        DESCRIPTION.

    '''
    x_3dim = np.zeros((x_df.shape[0]-sl+1,sl,x_df.shape[1]))
    y_3dim = np.zeros((y_df.shape[0]-sl+1,sl,ph))

    for i in range(x_df.shape[0]-sl+1):
            # TODO: if Tk is known, suppose is known
        x_3dim[i] = x_df[i: i+sl].values
        y_3dim[i] = y_df[i: i+sl].values
    return x_3dim, y_3dim

def data_pre(x, y, train_ratio, ph, sl):
    X_train = x[:int(x.shape[0] * train_ratio)]
    Y_train = y[:int(y.shape[0] * train_ratio)]
    X_test = x[int(x.shape[0] * train_ratio):]
    Y_test = y[int(y.shape[0] * train_ratio):]
    
    x_train, y_train = get_dataset(X_train,Y_train, sl, ph, target_co)
    x_test, y_test = get_dataset(X_test, Y_test, sl, ph, target_co)
    # x_train = torch.tensor(x_train, dtype=torch.float)
    # y_train = torch.tensor(y_train, dtype=torch.float)
    # x_test = torch.tensor(x_test, dtype=torch.float)
    # y_test = torch.tensor(y_test, dtype=torch.float)
    
    return x_train, y_train, x_test, y_test

class RegLSTM(nn.Module):
    def __init__(self, inp_dim, out_dim, mid_dim, mid_layers,batch):
        super(RegLSTM, self).__init__()
 
        self.rnn = nn.LSTM(inp_dim, mid_dim, mid_layers,batch_first=batch)
        self.reg = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, out_dim),
        )  # regression
 
    def forward(self, x):
        y = self.rnn(x)[0]  # y, (h, c) = self.rnn(x)
 
        batch_size, seq_len, hid_dim = y.shape
        y = y.reshape(-1, hid_dim)
        y = self.reg(y)
        y = y.reshape(batch_size, seq_len, -1)
        return y

def train_RNN(x_train,y_train):
    batch_size=20
    time_step=y_train.shape[1]
    epoch=500
    input_size=x_train.shape[-1]
    output_size=y_train.shape[-1]
    mid_dim=5
    mid_layers=256
    lr = 1e-4
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net=RegLSTM(input_size,output_size,mid_dim,mid_layers,True).to(device)
    criterion=nn.MSELoss()
    optimizer=torch.optim.Adam(net.parameters(),lr=lr)
    
    for i in range(epoch):
        for j in range(int((x_train.shape[0]-time_step+1)/batch_size)):
            train_X=x_train[j*batch_size:(j+1)*batch_size,:,:]
            train_Y=y_train[j*batch_size:(j+1)*batch_size,:,:]
            data_x=torch.tensor(train_X,dtype=torch.float32,device=device)
            data_y=torch.tensor(train_Y,dtype=torch.float32,device=device)
            out = net(data_x)
            loss=criterion(out,data_y)
            #loss = criterion(out[:,-1,:], var_y[:,-1,:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_X=x_train[(j+1)*batch_size:,:,:]
        train_Y=y_train[(j+1)*batch_size:,:,:]
        data_x=torch.tensor(train_X,dtype=torch.float32,device=device)
        data_y=torch.tensor(train_Y,dtype=torch.float32,device=device)
        out = net(data_x)
        loss = criterion(out, data_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%10==0:
            print('Epoch: {:4}, Loss: {:.5f}'.format(i, loss.item()))
    return net


def pred(x_df, net):
    pred=list()
    time_step=y_train.shape[1]
    input_size=x_train.shape[-1]

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(x_df.shape[0]):
        x_data = x.values[i:time_step+i,:].reshape(1,time_step,input_size)

        x_data=torch.tensor(x_data,dtype=torch.float32,device=device)
        tem=net(x_data).detach().numpy()
        pred.append(tem[0][-1])
    return 

def pred_and_analysis(x_train, y_train, x_test, y_test, net):
    # predict
    plt.plot(y_train, label = 'Ground Truth')
    plt.plot(train_pred, label = 'prediction', linestyle = '--')
    plt.title('CV_RMSE is {}'.format(cv_rmse(y_train.numpy(), train_pred)))
    plt.xlabel('TimeIndex')
    plt.ylabel('Temperature[C]')
    plt.legend()
    plt.show()
    
    plt.plot(y_train.numpy() - train_pred, label = 'Error')
    plt.xlabel('TimeIndex')
    plt.ylabel('Error[C]')
    plt.legend()
    plt.show()
    
    plt.plot(y_test, label = 'Ground Truth')
    plt.plot(test_pred, label = 'prediction', linestyle = '--')
    plt.title('CV_RMSE is {}'.format(cv_rmse(y_test.numpy(), test_pred)))
    plt.xlabel('TimeIndex')
    plt.ylabel('Temperature[C]')
    plt.legend()
    plt.show()
    
    plt.plot(y_test.numpy() - test_pred, label = 'Error')
    plt.xlabel('TimeIndex')
    plt.ylabel('Error[C]')
    plt.legend()
    plt.show()

    
    r2(y_test.numpy(), test_pred)

def cv_rmse(y,yhat):
    return np.sqrt(mse(y,yhat))/np.mean(y)

#%%
if __name__ == "__main__":
    original_data = load_data()
    cos = ['TimeIndex', 'mass_flow', 'T_oa', 'T_roo', 'GHI', 'Qint1', 'Qint2', 'Qint3']
    # cos = ['TimeIndex', 'T_oa', 'T_roo']

    data = original_data[cos]
    target_co = 'T_roo'
    his_l = 1*4
    ph = 1
    x, y = feature_engneering(data, his_l, target_co, ph)
    sl = 4 # sequence lenghth
    bs = 24 # batch size
    input_size = x.shape[1] -1
    output_size = sl
    
#%%
    # # #RNN
    train_ratio = 0.8
    x_train, y_train, x_test, y_test = data_pre(x, y, train_ratio, ph, sl)
    net = train_RNN(x_train, y_train)
    
    
    # rnn = train_RNN(train_loader, test_loader)
