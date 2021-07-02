from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_one_ep(num_zone = 1, history_Z_T = None, history_Env_T = None, fig_path_name = './simulation.png'):

    history_Z_T = history_Z_T#[:,0:180]
    history_Env_T = history_Env_T#[:,0:180]
    
    T_out = []
    T_zone = [[] for nz in range(num_zone)]
    
    T_out.extend(history_Env_T - 273.15)

    T_zone[0].extend(history_Z_T - 273.15)

    t = range(len(T_out))
    T_up = 26.0*np.ones([len(T_out)])
    T_low = 22.0*np.ones([len(T_out)])

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

if __name__ == "__main__":
    l = np.load('./experiments_results/his_obs.npy', allow_pickle=True)
    l_indoor = l[:, 1]
    l_outdoor = l[:, 2]
    plot_one_ep(num_zone = 1, history_Z_T = l_indoor, history_Env_T = l_outdoor, fig_path_name = './simulation.png')
    print(l[0:20])