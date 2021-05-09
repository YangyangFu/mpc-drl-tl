import pandas as pd
import numpy as np
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt

measurement_df = pd.read_csv('drl_measurement.csv')
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
        T_upper = np.array([30.0 for i in range(24)])
        T_upper[6:19] = 26.0
        T_lower = np.array([12.0 for i in range(24)])
        T_lower[6:19] = 22.0

        overshoot = []
        undershoot = []
        for k in range(number_zone):
            overshoot.append(np.array([float((TZone[i] -273.15) - T_upper[hindex]), 0.0]).max())
            undershoot.append(np.array([float(T_lower[hindex] - (TZone[i]-273.15)), 0.0]).max())

        penalty.append(alpha_up*sum(np.array(overshoot)) + alpha_low*sum(np.array(undershoot)))
    
    # sum up for rewards
    rewards = np.array(energy_cost) + np.array(penalty)

    return -rewards


rewards = get_rewards(measurement_df['PTot'],measurement_df['TRoo'],price_tou)
print rewards
nepoches = 20
nsteps = 96*7

acc_rewards = []
acc_period = 1  # accumative period in steps: 1 day
npoints = (nepoches*nsteps)//acc_period

acc_reward = 0
for i in range(npoints):
    acc_reward += np.array(rewards[i*acc_period:(i+1)*acc_period]).sum()
    acc_rewards.append(acc_reward)

plt.figure()
plt.plot(acc_rewards, 'r-')
plt.grid(True)
plt.savefig('rewards.pdf')
