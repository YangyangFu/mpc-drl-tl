# class of Q network
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import matlab.engine
import random

class NeuralNet(nn.Module):
    def __init__(self,layout,learning):
    # layout: architecture of Q network, describe number of neurons in each layer
    # learning: learning rate
        super(NeuralNet, self).__init__()
        self.layers = layout
        self.num_layers = len(self.layers)
        self.learning_rate = learning
        self.ave_value = []
        
        
        self.fc = nn.ModuleList()
        for nl in range(self.num_layers - 1):
            l = nn.Linear(self.layers[nl], self.layers[nl + 1])
            self.fc.append(l)
        #self.gemfield = nn.ModuleList([conv1, pool, conv2, fc1, fc2, fc3])
            
        
        self.loss_function = torch.nn.MSELoss()
        
    def forward(self, x):
        for nl in range(self.num_layers-1):
            if nl<self.num_layers-2:
                x = F.relu(self.fc[nl](x))
            else:
                x = self.fc[nl](x)
        return x
        
        
    # initialize variables in network
    def initialize(self):
        pass

    # close session of network
    def close(self):
        pass

    # train neural network with a batch data
    def train(self,optimizer, batch_x,batch_y):
    # batch_x: batch input, [[],...,[]]
    # batch_y: batch label, [[],...,[]]
        #optimizer = torch.optim.RMSProp(self.parameters(), lr = self.learning_rate)
        pre = self.forward(batch_x)
        loss = self.loss_function(pre, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #self.sess.run(self.train_step, feed_dict={self.x: batch_x, self.y_: batch_y})

    # save parameters to file
    def saving(self,path,Qname,Vname):
    # path: indicate the path from which to restore network
        try:
            torch.save(self.state_dict(), path+'/'+Qname)
            #save_path = self.saver.save(self.sess, path+'/'+Qname)
            np.save(path+'/'+Vname,[self.ave_value])
            # print("Model saved in file: %s" % save_path)
        except:
            print('error when saving variables!')
    
    # restore parameters from trained network
    def recover(self,path,Qname,Vname):
    # path: indicate the path from which to restore network
        try:
            # recover neural network parameters
            #self.saver.restore(self.sess, path+'/'+Qname)
            self.load_state_dict(torch.load(path+'/'+Qname))
            # recover average action value
            recall = np.load(path+'/'+Vname)
            # convert numpy array to list
            self.ave_value = recall[0].tolist()
            print('Model restored from file: %s' % path)
        except:
             print('error when restoring variables!')
