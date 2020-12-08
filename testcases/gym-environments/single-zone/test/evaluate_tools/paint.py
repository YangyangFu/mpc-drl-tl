import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_one_ep(num_zone = 1, history_Z_T = None, history_Env_T = None, ep = 1, fig_path_name = './simulation.png'):

    print("simulation history steps: ", history_Z_T.shape[0])
    
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