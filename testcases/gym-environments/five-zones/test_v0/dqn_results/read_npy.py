import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_one_ep(num_zone = 1, history_Z_T = None, history_Env_T = None, fig_path_name = './simulation.pdf'):

    history_Z_T = history_Z_T#[:,0:180]
    history_Env_T = history_Env_T#[:,0:180]
    
    T_out = []
    T_zone = [[] for nz in range(num_zone)]
    
    T_out.extend(history_Env_T - 273.15)

    for nz in range(num_zone):
        T_zone[nz].extend(history_Z_T.T[nz] - 273.15)

    t = range(len(T_out))
    T_up = 24.5*np.ones([len(T_out)])
    T_low = 23.5*np.ones([len(T_out)])

    T_up = [30.0 for i in range(len(T_out))]
    T_low = [12.0 for i in range(len(T_out))]

    for i in range(7):
        for j in range((19-7)*4):
            T_up[i*24*4 + (j) + 4*7] = 24.5
            T_low[i*24*4 + (j) + 4*7] = 23.5

    colors = ['green', 'black', 'magenta', 'brown', 'orange']
    labels = ['South','East','North','West','Core']
    # plt.figure(figsize=(10,20))
    # for nz in range(num_zone):
    #     temp=511+nz
    #     plt.subplot(temp)
    #     plt.plot(t,T_out,'b')
    #     plt.plot(t,T_zone[nz],color=colors[nz],linewidth=0.5)
    #     plt.plot(t,T_up,'r',t,T_low,'r')
    #     plt.axis([0,len(T_out),22,26])
    #     plt.xlabel('Simulation step')
    #     plt.ylabel('Temperature')
    #     plt.grid()
    # plt.show()
    # plt.savefig(fig_path_name)

    plt.figure()
    plt.plot(t,T_out,'b')
    for nz in range(num_zone):
        plt.plot(t,T_zone[nz],color=colors[nz],linewidth=0.5,label=labels[nz])
    plt.plot(t,T_up,'r',t,T_low,'r')
    plt.axis([0,len(T_out),22,26])
    plt.xlabel('Simulation step')
    plt.ylabel('Temperature')
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(fig_path_name)


if __name__ == "__main__":
    l = np.load('./his_obs.npy', allow_pickle=True)
    l_indoor = l[:, 1:6]
    l_outdoor = l[:, 6]
    plot_one_ep(num_zone = 5, history_Z_T = l_indoor, history_Env_T = l_outdoor, fig_path_name = './simulation.pdf')

    l = np.load('./his_rew.npy', allow_pickle=True)
    print(l[200:220])