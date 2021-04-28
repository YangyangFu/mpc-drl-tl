import numpy as np
from coarse_models import Zone

# define a zone
L = 4 # this has to base on the training results
json_file = "zone.json"
zone1 = Zone(L=L, json_file=json_file)

# predict future temperature given inputs
T_his_meas = np.array([289.6985339, 289.5517624, 289.4054251, 289.2596627])
T_his_pred = np.array([289.72356823, 289.57173601, 289.39570684, 289.217086])
mz = 0.075
Toa = 290.15
Tnext = zone1.predict(Tz_his_meas = T_his_meas,
                Tz_his_pred = T_his_pred,
                mz = mz,
                Toa = Toa)
print Tnext
