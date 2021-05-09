import numpy as np 
import pandas as pd
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
# load testbed
from pyfmi import load_fmu

def interpolate_dataframe(df,new_index):
    """Interpolate a dataframe along its index based on a new index
    """
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name

    for col_name, col in df.items():
        df_out[col_name] = np.interp(new_index, df.index, col)    
    return df_out

mass_flow_nominal = 0.75 # kg/s

start = 212*24*3600.
period = 7*24*3600.
end = start+period
dt = 15*60.

# actions from DRL
action = np.load('history_Action.npy')
print action
print action.shape

# 20 epoch
actions = []
nepoches = int(action.shape[0])
nsteps = int(action.shape[1])
nzones = int(action.shape[2])

# for zone 1
for epoch in range(nepoches):
    actions += list(action[epoch,:,0])
print actions

# transform to damper position
u_opt = [action/10. for action in actions]

# load FMU - set up measumrents
measurement_names = ['time','TRoo', 'PTot']

# 20 epoch continuously simulated in FMU
res=[]
for epoch in range(nepoches):
    #do one week simulation

    hvac = load_fmu('SingleZoneDamperControl.fmu')
    options = hvac.simulate_options()
    options['ncp'] = 500.
    options['initialize'] = True
    options['filter'] = measurement_names
    options['result_handling'] = 'memory'

    # main loop - do step
    t = start
    i = 0
    while t < end:
        u = u_opt[epoch*nsteps+i]
        u_traj = np.transpose(np.vstack(([t,t+dt],[u,u])))
        input_object = ("uFan",u_traj)
        ires = hvac.simulate(start_time = t,
                    final_time = t+dt, 
                    options = options,
                    input = input_object)
        
        res.append(ires)

        t += dt 
        i += 1
        options['initialize'] = False

# get all the measurements by epoch
measurements={}
for epoch in range(nepoches):
    index_from = epoch*nsteps
    index_end = (epoch+1)*nsteps

    res_epoch = res[index_from:index_end]

    # read measurements from results
    measurements_epoch = {}
    for name in measurement_names:
        value_name=[]
        for ires in res_epoch:
            value_name += list(ires[name])
        measurements_epoch[name] = value_name

    measurements[epoch] = measurements_epoch

# interpolate data for each epoch and then combine all the results in series - ignoring time index
t_intp = np.arange(start, end, dt)
measurement_df=pd.DataFrame()

for epoch in range(nepoches):
    measurement_epoch = measurements[epoch]
    measuremnt_epoch_df = pd.DataFrame(measurement_epoch,index=measurement_epoch['time'])
    measurement_epoch_df_intp = interpolate_dataframe(measuremnt_epoch_df,t_intp)
    measurement_epoch_df_intp['time'] = measurement_epoch_df_intp.index 
    measurement_df = pd.concat([measurement_df,measurement_epoch_df_intp], ignore_index=True)

print measurement_df
measurement_df.to_csv('drl_measurement.csv')

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

acc_rewards = []
acc_period = 96 # accumative period in steps: 1 day
npoints = (nepoches*nsteps)//acc_period

acc_reward = 0
for i in range(npoints):
    acc_reward += np.array(rewards[i*acc_period:(i+1)*acc_period]).sum()
    acc_rewards.append(acc_reward)

plt.figure()
plt.plot(acc_rewards,'r-')
plt.grid(True)
plt.savefig('rewards.pdf')


# epoch by epoch: 20 points 
# 1 day by 1 day - 140 points
