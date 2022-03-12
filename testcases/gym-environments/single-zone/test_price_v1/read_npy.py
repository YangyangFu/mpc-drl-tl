import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
sub_dirs = []
tuning_dir = os.path.join(cur_dir, 'ddqn_tuning')
for it in os.scandir(tuning_dir):
    if it.is_dir():
        sub_dirs.append(it.path)

for sub_dir in sub_dirs:
    data_path = os.path.join(sub_dir, 'log_ddqn', 'JModelicaCSSingleZoneEnv-price-v1')
    print(data_path)
    data_path = '\\\\?\\'+data_path # troubleshoot long path issues in windows os
    acts = np.load(os.path.join(data_path, 'his_act.npy'), allow_pickle=True)
    obss = np.load(os.path.join(data_path, 'his_obs.npy'), allow_pickle=True)
    rews = np.load(os.path.join(data_path, 'his_rew.npy'), allow_pickle=True)

    TRoo_obs = [T- 273.15 for T in obss[:-1, 1]]
    TOut_obs = [T- 273.15 for T in obss[:-1, 2]]

    t = range(len(TRoo_obs))
    ndays = int(len(TRoo_obs)//96.)
    print("we have "+ str(ndays) +" days of data)")
    T_up = 26.0*np.ones([len(TRoo_obs)])
    T_low = 22.0*np.ones([len(TRoo_obs)])

    T_up = [30.0 for i in range(len(TRoo_obs))]
    T_low = [12.0 for i in range(len(TRoo_obs))]
    for i in range(ndays):
        for j in range((19-8)*4):
            T_up[i*24*4 + (j) + 4*7] = 26.0
            T_low[i*24*4 + (j) + 4*7] = 22.0

    plt.figure(figsize=(12,6))
    plt.subplot(211)
    plt.plot(t, TOut_obs, 'b', label="Outdoor")
    plt.plot(t,T_up,'r',t,T_low,'r')
    plt.plot(t, TRoo_obs, 'k', label="Indoor")
    plt.ylim([10,40])
    plt.grid()
    plt.ylabel("Temperaure [C]")
    plt.xlabel("Time Step")

    plt.subplot(212)
    plt.plot(t, [acts[i]/50. for i in range(len(t))])
    plt.ylabel("Speed")
    plt.xlabel("Time Step")
    plt.savefig(os.path.join(sub_dir,"final_train.pdf"))
    plt.savefig(os.path.join(sub_dir,"final_train.png"))
    plt.close()

    print(acts)
