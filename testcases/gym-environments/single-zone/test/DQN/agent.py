# class of learning agent
import numpy as np
import torch
from neural import NeuralNet
from memory import Memory
import random


# softmax function
def softmax(data):
    return np.exp(data)/np.sum(np.exp(data))

class Agent(object):
    def __init__(self,layout,batch,explore,explore_l,explore_d,learning,decay,k,m_size,pa):
        self.layers = layout
        self.batch_size = batch
        self.decay_rate = decay
        self.learning_period = k                       # every k times updating, copy Q network
        self.learning_rate = learning
        self.num_action = self.layers[len(self.layers)-1]
        self.explore_low = explore_l
        self.explore_decay = explore_d
        self.directory = pa
        self.device = torch.device("cuda")
        #self.device = torch.device("cpu")

        self.explore_rate = explore 
        self.index_learning = 0                        # how many times updating Q network

        # reward scaling
        self.scaling = 800.0

        ##### build Q network
        self.Q = NeuralNet(self.layers,self.learning_rate).to(self.device)
        total = sum([param.nelement() for param in self.Q.parameters()])
        print('Number of params: ', (total))
        self.Q_copy = NeuralNet(self.layers,self.learning_rate).to(self.device)
        self.optimizer_Q = torch.optim.Adam(self.Q.parameters(), lr = self.Q.learning_rate)

        # build memory
        self.memory = Memory(m_size)
        self.min_replay = 100
        


    # initialize agent
    def initialize(self, path):
        self.Q.initialize()
        self.Q.saving(path,'Q_net.pt','Q_net.npy')
        self.Q_copy.recover(path,'Q_net.pt','Q_net.npy')
        self.Q_copy.saving(path,'Q_copy.pt','Q_copy.npy')

    # restore agent from trained data
    def recover(self,path):
        # module recovery
        self.Q.recover(path,'Q_net.pt','Q_net.npy');
        self.Q_copy.recover(path,'Q_copy.pt','Q_copy.npy')
        self.memory.recover(path,'memory.npy')
        # agent parameters recovery
        recall = np.load(path+'/agent.npy')
        self.explore_rate = recall[0]
        self.index_learning = recall[1]

    def backup(self,path):
        # module backup
        self.Q.saving(path,'Q_net.pt','Q_net.npy')
        self.Q_copy.saving(path,'Q_copy.pt','Q_copy.npy')
        self.memory.backup(path,'memory.npy')
        # agent parameters backup
        np.save(path+'/agent.npy',[self.explore_rate,self.index_learning])

    # select an action based on policy
    def policy(self,state):
    # state: current observed state
        exploration = np.random.choice(range(2),1,p=[1-self.explore_rate,self.explore_rate])
        # exploration==1: explore
        # exploration==0: exploit
        if exploration==1:          # exploration
            action_index = np.random.choice(range(self.num_action),1)[0]
            
            print('\r')
            print('              explore:  '+str(action_index))
            print('\r')
            
            with torch.no_grad():
                train_state = torch.from_numpy(np.asarray([state])).float()
                train_state = train_state.to(self.device)
                action_value = self.Q.forward(train_state).detach().cpu().numpy()
                action_index_nn = np.argmax(action_value[0])
                print(action_index_nn)
            
            return action_index
        else:                       # exploitation
            with torch.no_grad():
                train_state = torch.from_numpy(np.asarray([state])).float()
                train_state = train_state.to(self.device)
                action_value = self.Q.forward(train_state).detach().cpu().numpy()

                action_index_nn = np.argmax(action_value[0])

            print('\r')
            print('exploit:  '+str(action_index_nn))
            print('\r')

            return action_index_nn

    
    # store current observation to memory
    def memorize(self,observation):
    # observation: new observation (s,a,r,s')
        self.memory.add(observation)

    # replay memory
    def replay(self):
        transisions = self.memory.sampling(self.batch_size)      # data type: array of Matlab double
        # train Q network with mini-batch
        states = []
        action_values = []
        print('len_trans', len(transisions))
        
        for nt in range(len(transisions)):
            # construct training input
            states.append(transisions[nt][0])                    # data type: array of Matlab double

            # construct training label
            
            with torch.no_grad():
                train_transisions = torch.from_numpy(np.asarray([transisions[nt][0]])).float()
                train_transisions = train_transisions.to(self.device)
                target_value0 = self.Q.forward(train_transisions)[0]
                target_value = target_value0.detach().cpu().numpy()
                ### calculate target Q(s,a) based on Bellman equation
                train_transisions1 = torch.from_numpy(np.asarray([transisions[nt][3]])).float()
                train_transisions1 = train_transisions1.to(self.device)
                Q_copy_target_value0 = self.Q_copy.forward(train_transisions1)[0].cpu().numpy()
                target = (transisions[nt][2][0] + transisions[nt][2][1]) + self.decay_rate * max(Q_copy_target_value0)

                # clip target
                target_value[transisions[nt][1]] = self.clip(target)
                
                action_values.append(target_value)
        
        # perform gradient descent with mini-batch data
        train_state = torch.from_numpy(np.asarray(states)).float()
        train_state = train_state.to(self.device)
        train_action_values = torch.from_numpy(np.asarray(action_values)).float()
        train_action_values = train_action_values.to(self.device)
        self.Q.train(self.optimizer_Q, train_state, train_action_values)
        # update target Q network
        self.index_learning = self.index_learning + 1
        if self.index_learning>=self.learning_period:
            # copy Q network
            path = self.directory
            self.Q.saving(path,'Q_net.pt','Q_net.npy')
            self.Q_copy.recover(path,'Q_net.pt','Q_copy.npy')
            self.index_learning = 0
    
    # target value clipping
    def clip(self,target):
    # target: target value
        clipped_target = 0.0
        if target > 0.0:
            clipped_target = 0.0
        elif target < -1.0:
            clipped_target = -1.0
        else:
            clipped_target = target
        return clipped_target

    # learn one episode
    def learning(self,state,action_index,reward_list,state_prime):
        self.explore_rate = max(self.explore_rate - self.explore_decay, self.explore_low)

        # reward_list[0] cost, reward_list[1] violation
        # reward scaling   
        reward = [reward_list[0][0]/self.scaling, reward_list[1][0]/self.scaling]

        print('\r')
        print('index_learning: %d   room: %0.3f --> %0.3f   cost: %f   penalty: %f' %(self.index_learning,state[1]*(40-10)+10,state_prime[1]*(40-10)+10,reward[0],reward[1]))
                
        # add current transition pair into memory
        self.memorize([state,action_index,reward,state_prime])
        
        # train Q network when there are enough state transitions
        if self.memory.size>=self.min_replay:
            print('experience replay...')
            self.replay()
            print('\r')    
        else:
            print('not enough observations...')
    
            
            
