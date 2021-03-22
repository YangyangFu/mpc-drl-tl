from sklearn.externals import joblib    
import numpy as np

def zone_temperature(Tz_his, mz, Toa):
    ann = joblib.load('ann.pkl')
    x=list(Tz_his)
    x.append(mz)
    x.append(Toa)
    x = np.array(x).reshape(1,-1)
    print x
    y=ann.predict(x)
    y = max(273.15+14,min(y,273.15+35))
    return y

print zone_temperature([293.15, 293.15, 294.15, 295.15], 0.45, 292.15)