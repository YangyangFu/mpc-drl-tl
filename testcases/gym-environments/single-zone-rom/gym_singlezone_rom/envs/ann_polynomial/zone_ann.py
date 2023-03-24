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
import seaborn as sns
from torch.autograd import Variable
#%%
def load_data():
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
        df[f"{col}_std_lag{window}"] = lag_std[col]
    return df



def feature_engneering(data):
    df_pre = data.copy()
    df_pre['T_roo'] -= 273.15
    df_pre['T_oa'] -= 273.15
    df_pre['T_sa'] -= 273.15

    lz=4
    lo=4
    zone_temp_name = "T_roo"
    Tz_his = pd.DataFrame(df_pre[['TimeIndex',zone_temp_name]])
    for i in range(lz):
        # if i <= 5:
        #     continue
        Tz_his['Tz_'+str(i+1)] = Tz_his[zone_temp_name].values
        shift = Tz_his['Tz_'+str(i+1)].shift(periods=i+1)
        Tz_his['Tz_'+str(i+1)]=shift.values
    Tz_his=Tz_his.drop(columns=[zone_temp_name])
    df_pre = pd.merge(df_pre, Tz_his, on='TimeIndex')

    ph = 4
    oa_name = 'T_oa'
    To_his = pd.DataFrame(df_pre[['TimeIndex', oa_name]])
    for i in range(lo):
        To_his['To_'+str(-(i+1))] = To_his[oa_name].values
        shift = To_his['To_'+str(-(i+1))].shift(periods=(i+1))
        To_his['To_'+str(-(i+1))]=shift.values
    To_his=To_his.drop(columns=[oa_name])
    df_pre = pd.merge(df_pre, To_his, on='TimeIndex')
    
    To_fut = pd.DataFrame(df_pre[['TimeIndex', oa_name]])
    for i in range(lo):
        To_fut['To_'+str(i+1)] = To_fut[oa_name].values
        shift = To_fut['To_'+str(i+1)].shift(periods=-(i+1))
        To_fut['To_'+str(i+1)]=shift.values
    To_fut=To_fut.drop(columns=[oa_name])
    df_pre = pd.merge(df_pre, To_fut, on='TimeIndex')

    df_pre = lag_feature(df_pre)
    df_pre.dropna(how = 'any', inplace = True)
    
    target_co = 'T_roo'
    T_fut = pd.DataFrame(df_pre[target_co])
    for i in range(ph):
        T_fut['Tz_'+str(i+1)] = T_fut[target_co].values
        shift = T_fut['Tz_'+str(i+1)].shift(periods=-(i+1))
        T_fut['Tz_'+str(i+1)]=shift.values
    T_fut.drop(target_co, axis = 1, inplace = True)
    T_fut.dropna(how = 'any', inplace = True)
    
    df_pre = df_pre[:T_fut.shape[0]]
    
    # TODO solar radiation previous and future values

    return df_pre, T_fut


def split_data(df_pre, T_fur):
    x_cos = [

                    'T_roo',
                    'T_oa',
                    'GHI',
                    'mass_flow',
               # 'GHI',  'Qint1', 'Qint2', 'Qint3',
               #     'Mret', 'Hinfi', 'Hinfo', 'Minf',
                   # 'Qint1', 'Qint2', 'Qint3',
                   # 'Mret', 'Hsup', 'Hret', 'Hinfi',
                   # 'Hinfo', 'Minf',
                    'Tz_4', 'Tz_3', 'Tz_2', 'Tz_1',
                   'To_-4', 'To_-3', 'To_-2','To_-1',
                   'To_1', 'To_2', 'To_3', 'To_4',
                   
                 #  'To_1', 'To_2', 'To_3',
                 # 'To_4'
               # 'T_oa_mean_lag4', 'T_oa_max_lag4', 'T_oa_min_lag4',
               # 'T_oa_std_lag4'
               # , 'T_roo_mean_lag4', 'T_roo_max_lag4', 'T_roo_min_lag4',
               # 'T_roo_std_lag4'
                   # 'To_1', 'To_2', 'To_3', 'To_4',

             ]
    # y_co = ['T_roo']
    df_train = df_pre[:int(df_pre.shape[0] * 0.8)]
    df_test = df_pre[int(df_pre.shape[0] * 0.8):]
    df_x_train = df_train[x_cos]
    # df_y_train = df_train[y_co]
    df_y_train = T_fur[:int(df_pre.shape[0] * 0.8)]
    df_x_test = df_test[x_cos]
    # df_y_test = df_test[y_co]
    df_y_test = T_fur[int(df_pre.shape[0] * 0.8):]


    return df_x_train, df_y_train, df_x_test, df_y_test

def to_ann(df_x_train, df_y_train, df_x_test, df_y_test):
    x_train = torch.tensor(df_x_train.values, dtype=torch.float)
    y_train = torch.tensor(df_y_train.values, dtype=torch.float)
    x_test = torch.tensor(df_x_test.values, dtype=torch.float)
    y_test = torch.tensor(df_y_test.values, dtype=torch.float)
    return x_train, y_train, x_test, y_test

def to_3_dimension(df, sl = 4):
    '''
    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    sl : TYPE
        sequence length.

    Returns
    -------
    df_3 : TYPE
        DESCRIPTION.

    '''
    dim3 = np.zeros((df.shape[0]-sl+1,sl,df.shape[1]))
    for i in range(df.shape[0]-sl+1):
        # if i+sl <= df.shape[0]:
        dim3[i] = df[i: i+sl].values
    return dim3
def to_rnn(df_x_train, df_y_train, df_x_test, df_y_test
           , sl, bs, input_size, output_size):
    
    x_train = to_3_dimension(df_x_train, sl)

    y_train = df_y_train.copy().values[:-sl+1]

    x_test = to_3_dimension(df_x_test, sl)

    y_test = df_y_test.copy().values[:-sl+1]

    x_train = torch.tensor(x_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)
    x_test = torch.tensor(x_test, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.float)
    
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    batch_size = x_train.shape[1]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

# Define neural network architecture
class Net(nn.Module):
    def __init__(self, features):
        super(Net, self).__init__()
        # self.fc1 = nn.Linear(features, 10)
        # self.fc2 = nn.Linear(10, 1)
        self.linear_relu1 = nn.Linear(features, 256)
        self.linear_relu2 = nn.Linear(256, 256)
        self.linear_relu3 = nn.Linear(256, 256)
        self.linear_relu4 = nn.Linear(256, 256)
        self.linear_relu5 = nn.Linear(256, 256)
        self.linear_relu6 = nn.Linear(256, 256)

        self.linear7 = nn.Linear(256, 4)
        # self.activation = nn.ReLU()

    def forward(self, x):
        y_pred = self.linear_relu1(x)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu2(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu3(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu4(y_pred)
        y_pred = nn.functional.relu(y_pred)
        
        y_pred = self.linear_relu5(y_pred)
        y_pred = nn.functional.relu(y_pred)
        y_pred = self.linear_relu6(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear7(y_pred)
        
        
        
        # x = self.activation(self.linear_relu1(x))
        # x = self.activation(self.linear_relu2(x))
        # x = self.activation(self.linear_relu3(x))
        # x = self.activation(self.linear_relu4(x))
        # x = self.activation(self.linear_relu5(x))
        # x = self.activation(self.linear6(x))
        
        return y_pred


# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(RNN, self).__init__()

#         self.hidden_size = hidden_size
#         self.rnn = nn.RNN(input_size, hidden_size)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x, h=None):
#         out, h = self.rnn(x, h)
#         out = self.fc(out[-1])
#         return out, h
    
# Train the model for 1000 epochs
def train_NN( x_train,y_train):
    net = Net(features=x_train.shape[1])
    # Define the loss function and optimizer
    criterion = nn.MSELoss(reduction='mean')
    # optimizer = optim.SGD(net.parameters(), lr=0.1)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)

    for epoch in range(2000):
        # Zero the gradients
        optimizer.zero_grad()
    
        # Forward pass
        y_pred = net(x_train)
        loss = criterion(y_pred, y_train)
        
        if torch.isnan(loss):
            break
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # # Backward pass
        # loss.backward()
        # optimizer.step()
    
        # Print the loss every 10 epochs
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 10, loss.item()))
    return net

def train_RNN(x_train,y_train, train_loader):
    features = x_train.shape[1]
    batch_size=32
    time_step=1
    epoch=500
    input_size=features
    output_size=1
    hidden_size=8
    mid_layers=2
    
    class RNN(nn.Module):
        def __init__(self,input_size):
            super(RNN, self).__init__()
    
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=32,
                batch_first=True,
                num_layers=1,
            )
    
            self.out = nn.Linear(32, 1)
    
        def forward(self, x, h_state):
            r_out, h_state = self.rnn(x, h_state)
            outs = []
            for time_step in range(r_out.size(0)):
                outs.append(self.out(r_out[time_step, :, :]))
    
            return torch.stack(outs, dim=1), h_state

    # class RNN(nn.Module):
    #     # def __init__(self, input_size, hidden_size, output_size):
    #     #     super(LSTMModel, self).__init__()
    #     #     self.hidden_size = hidden_size
    #     #     self.lstm = nn.LSTM(input_size, hidden_size)
    #     #     self.fc = nn.Linear(hidden_size, output_size)
    
    #     # def forward(self, x):
    #     #     batch_size = x.size(1)
    #     #     h0 = torch.zeros(1, batch_size, self.hidden_size)
    #     #     c0 = torch.zeros(1, batch_size, self.hidden_size)
    #     #     output, (hn, cn) = self.lstm(x, (h0, c0))
    #     #     output = self.fc(output[-1])
    #     #     return output
    #     def __init__(self, feature):
    #         super(RNN, self).__init__()
    
    #         self.rnn = nn.RNN(  # 这回一个普通的 RNN 就能胜任
    #             input_size=feature,
    #             hidden_size=32,     # rnn hidden unit
    #             num_layers=1,       # 有几层 RNN layers
    #             batch_first=True,   # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
    #         )
    #         self.out = nn.Linear(32, 1)
    
    #     def forward(self, x, h_state):  # 因为 hidden state 是连续的, 所以我们要一直传递这一个 state
    #         # x (batch, time_step, input_size)
    #         # h_state (n_layers, batch, hidden_size)
    #         # r_out (batch, time_step, output_size)
    #         r_out, h_state = self.rnn(x, h_state)   # h_state 也要作为 RNN 的一个输入
    
    #         outs = []    # 保存所有时间点的预测值
    #         for time_step in range(r_out.size(1)):    # 对每一个时间点计算 output
    #             outs.append(self.out(r_out[:, time_step, :]))
    #         return torch.stack(outs, dim=1), h_state
        
        
    rnn = RNN(input_size)
    LR = 0.01 #learning rate
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all parameters
    criterion = nn.MSELoss(reduction='mean')
    h_state = None
    epochs= 100
    print(rnn)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs, h_state = rnn(inputs,h_state)
            h_state = h_state.data
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, epochs, running_loss/len(train_loader)))

    return rnn

    
def pred_analysis(test_pred, train_pred):
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
    plt.title('RMSE is {}'.format(rmse(y_test.numpy(), test_pred)))
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
    
def sequence_visualize(pred, truth):
    for i in range(pred.shape[0]):
        plt.plot(truth[i], label = 'Ground Truth')
        plt.plot(pred[i], label = 'prediction', linestyle = '--')
        plt.title('RMSE is {}'.format(rmse(truth[i].numpy(), pred[i])))
        plt.xlabel('TimeIndex')
        plt.ylabel('Temperature[C]')
        plt.legend()
        plt.show()
        


def previous6(df_pre):
    df_test = df_pre[int(df_pre.shape[0] * 0.8):]
    return df_test['Tz_7'].values

def cv_rmse(y,yhat):
    return np.sqrt(mse(y,yhat))/np.mean(y)

def rmse(y,yhat):
    return np.sqrt(mse(y,yhat))

def cal_error(pred, truth):
    cv_rmses = []
    for i in range(pred.shape[1]):
        cv_rmses.append(cv_rmse(truth[:,i].numpy(),pred[:,i]))
        errors = truth[:,i].numpy()-pred[:,i]
        sns.distplot(errors)
        # sns.histplot(errors)
        plt.title('predtion of step {}'.format(i))
        plt.xlabel('Absolute error')
        plt.show()
#%%
if __name__ == "__main__":
    data = load_data()
    df_pre, T_fut = feature_engneering(data)
    df_x_train, df_y_train, df_x_test, df_y_test = split_data(df_pre, T_fut)
    #%% ANN
    x_train, y_train, x_test, y_test = to_ann(df_x_train, df_y_train, df_x_test, df_y_test)
    # Create a dataset of 100 samples with 1 feature and 1 target value
    #%%
    ann = train_NN(x_train, y_train)

    # #save
    torch.save(ann.state_dict(), '..//zone_ann.pt')
    # torch.save(ann, '..//zone_ann.pt')
    # #load
    # ann = torch.load('..//zone_ann.pt')
    
    
    # # #predcit
    test_pred = ann(x_test).detach().numpy()
    train_pred = ann(x_train).detach().numpy()
    # sequence_visualize(test_pred, y_test)
    # sequence_visualize(train_pred, y_train)

    cal_error(train_pred, y_train)
    cal_error(test_pred, y_test)

    # # predictions =  previous6(df_pre)
    # # predictions = rnn(x_test).detach().numpy()
    
#%%
    # #RNN
    # sl = 4 # sequence lenghth
    # bs = 24 # batch size
    # input_size = df_x_train.shape[1]
    # output_size = df_y_train.shape[1]
    
    # train_loader, test_loader = to_rnn(df_x_train, df_y_train, df_x_test, df_y_test
    #                                    , sl, bs, input_size, output_size)
    
    
    
    # rnn = train_RNN(train_loader, test_loader)
