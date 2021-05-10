import pandas as pd
import numpy as np
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt

# some general settings
nepochs = 50
ts = 212*24*3600.
ndays = 7
te = ts + ndays*24.3600
dt = 15*60.
nsteps = int(24*3600./dt)*ndays

# Read baseline and MPC measurement for calculating DRL rewards
measurement_base = pd.read_csv('measurement_base.csv')
measurement_mpc = pd.read_csv('measurement_mpc.csv')

# calculate rewards
price_tou = [0.0640, 0.0640, 0.0640, 0.0640, 
    0.0640, 0.0640, 0.0640, 0.0640, 
    0.1391, 0.1391, 0.1391, 0.1391, 
    0.3548, 0.3548, 0.3548, 0.3548, 
    0.3548, 0.3548, 0.1391, 0.1391, 
    0.1391, 0.1391, 0.1391, 0.0640]

def get_rewards(Ptot,TZone,price_tou):
    n= len(Ptot)
    energy_cost = []
    penalty = []

    alpha_up = 200
    alpha_low = 200

    for i in range(n):
        # assume 1 step is 15 minutes and data starts from hour 0
        hindex = (i%96)//4
        power=Ptot[i]
        price = price_tou[hindex]

        energy_cost.append(power/1000./4*price)

        # zone temperature penalty
        number_zone = 1

        # zone temperature bounds - need check with the high-fidelty model
        T_upper = np.array([30.0 for j in range(24)])
        T_upper[7:19] = 26.0
        T_lower = np.array([12.0 for j in range(24)])
        T_lower[7:19] = 22.0

        overshoot = []
        undershoot = []
        for k in range(number_zone):
            overshoot.append(np.array([float((TZone[i] -273.15) - T_upper[hindex]), 0.0]).max())
            undershoot.append(np.array([float(T_lower[hindex] - (TZone[i]-273.15)), 0.0]).max())

        penalty.append(alpha_up*sum(np.array(overshoot)) + alpha_low*sum(np.array(undershoot)))
    
    # sum up for rewards

    return -np.array([energy_cost,penalty]).transpose()

#### get rewards
#================================================================================
rewards_base = get_rewards(measurement_base['PTot'],measurement_base['TRoo'],price_tou)
rewards_mpc = get_rewards(measurement_mpc['PTot'],measurement_mpc['TRoo'],price_tou)

rewards_base = pd.DataFrame(rewards_base,columns=[['ene_cost','penalty']])
rewards_base['sum'] = rewards_base.sum(axis=1)
rewards_mpc = pd.DataFrame(rewards_mpc,columns=[['ene_cost','penalty']])
rewards_mpc['sum'] = rewards_mpc.sum(axis=1)

# get rewards - DRL
rewards_drl_hist = np.load('history_Reward.npy')
rewards_drl = []
# for zone 1
for epoch in range(nepochs):
    for step in range(nsteps):
        rewards_drl.append(list(rewards_drl_hist[epoch,step,0,:,0]))

rewards_drl = pd.DataFrame(np.array(rewards_drl),columns=[['ene_cost','penalty']])
rewards_drl['sum'] = rewards_drl.sum(axis=1)


# plot rewards with moving windows - epoch-by-epoch
moving = ndays*24*3600.//dt 
rewards_moving_base = rewards_base['sum'].groupby(rewards_base.index//moving).sum()
rewards_moving_mpc = rewards_mpc['sum'].groupby(rewards_mpc.index//moving).sum()
rewards_moving_drl = rewards_drl['sum'].groupby(rewards_drl.index//moving).sum()

plt.figure(figsize=(9,6))
plt.plot(list(rewards_moving_base.values)*nepochs,'b-',label='RBC')
plt.plot(list(rewards_moving_mpc.values)*nepochs,'b--',label='MPC')
plt.plot(rewards_moving_drl['sum'],'r--',label='DQN')
plt.ylabel('rewards')
plt.xlabel('epoch')
plt.grid(True)
plt.legend()
plt.savefig('rewards_moving_all.pdf')