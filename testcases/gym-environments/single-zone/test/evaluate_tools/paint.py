import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_one_ep(num_zone = 1, history_Z_T = None, history_Env_T = None, ep = 1, fig_path_name = './simulation.png'):

    print("simulation history steps: ", history_Z_T.shape[0])
    history_Z_T = history_Z_T#[:,0:180]
    history_Env_T = history_Env_T#[:,0:180]
    ne = history_Z_T.shape[0]-ep
    
    T_out = []
    T_zone = [[] for nz in range(num_zone)]
    
    T_out.extend(history_Env_T[ne])
    for nz in range(num_zone):
        T_zone[nz].extend(history_Z_T[ne,:,nz])

    t = range(len(T_out))
    T_up = 24.0*np.ones([len(T_out)])
    T_low = 19.0*np.ones([len(T_out)])

    colors = [[121/255.0,90/255.0,206/255.0],
        [91/255.0,131/255.0,28/255.0],
        [109/255.0,70/255.0,160/255.0],
        [18/255.0,106/255.0,118/255.0],
        [0/255.0,0/255.0,0/255.0]]
    plt.figure()

    plt.plot(t,T_out,'b')
    
    for nz in range(num_zone):
        plt.plot(t,T_zone[nz],color=colors[nz])
    plt.plot(t,T_up,'r',t,T_low,'r')
    plt.axis([0,len(T_out),10,40])
    plt.xlabel('Simulation step')
    plt.ylabel('Temperature')
    plt.grid()
    plt.show()
    plt.savefig(fig_path_name)


def getViolation(num_zone = 1, ep = 1, history_Z_T = None, delCtrl = 15*60.0, num_days = 31):
    ne = history_Z_T.shape[0] - ep
    T_zone = [[] for i in range(num_zone)]
    for nz in range(num_zone):
        T_zone[nz].extend(history_Z_T[ne,:,nz])  
    len_his = len(T_zone[0])
    T_up = 24*np.ones([len_his])
    T_low = 19*np.ones([len_his])
    violation = [[] for i in range(num_zone)]
    num_step_hour = 3600 / delCtrl
    num_step_day = 24*3600 / delCtrl
    
    max_violation = [0.0 for nz in range(num_zone)]

    for ns in range(int(num_step_day*num_days)):
        t = ns*3600/num_step_hour
        t = int((t%86400)/3600)
        for nz in range(num_zone):
            if T_zone[nz][ns]>T_up[t]:# or T_zone[nz][ns]<T_low[t]:
                max_violation[nz] = max(max_violation[nz],max(max(T_zone[nz][ns]-T_up[t],0.0),max(T_low[t]-T_zone[nz][ns],0.0)))
                violation[nz].append(1)
            else:
                violation[nz].append(0)
    for nz in range(num_zone):
        print('zone '+str(nz+1)+': ', np.sum(violation[nz])/float(len(violation[nz]))*100.0,'%')
    print('max_violation: ', max_violation)


def plot_one_action_ep(num_zone = 1, history_Z_T = None, history_Env_T = None, history_action = None, ep = 1, fig_path_name = './simulation.png'):

    print("simulation history steps: ", history_Z_T.shape[0])
    history_Z_T = history_Z_T#[:,0:180]
    history_Env_T = history_Env_T#[:,0:180]
    #print(history_action)
    history_action = history_action + 19.0

    ne = history_Z_T.shape[0]-ep
    
    T_out = []
    T_zone = [[] for nz in range(num_zone)]
    
    T_out.extend(history_Env_T[ne])
    for nz in range(num_zone):
        T_zone[nz].extend(history_Z_T[ne,:,nz])

    t = range(len(T_out))
    T_up = 24.0*np.ones([len(T_out)])
    T_low = 19.0*np.ones([len(T_out)])

    colors = [[121/255.0,90/255.0,206/255.0],
        [91/255.0,131/255.0,28/255.0],
        [109/255.0,70/255.0,160/255.0],
        [18/255.0,106/255.0,118/255.0],
        [0/255.0,0/255.0,0/255.0]]
    plt.figure()

    plt.plot(t,T_out,'b')
    
    for nz in range(num_zone):
        plt.plot(t,T_zone[nz],color=colors[nz])
        plt.plot(t,history_action[0,:,nz],color=colors[nz+1])

    plt.plot(t,T_up,'r',t,T_low,'r')
    plt.axis([0,len(T_out),10,40])
    plt.xlabel('Simulation step')
    plt.ylabel('Temperature')
    plt.grid()
    plt.show()
    plt.savefig(fig_path_name)

    