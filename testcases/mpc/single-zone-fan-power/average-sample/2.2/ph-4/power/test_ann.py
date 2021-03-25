from sklearn.externals import joblib    
import numpy as np

def zone_temperature(Tz_his, mz, Toa):
    ann = joblib.load('ann.pkl')
    x=list(Tz_his)
    x.append(mz)
    x.append(Toa)
    x = np.array(x).reshape(1,-1)
    y=float(ann.predict(x))
    
    return np.maximum(273.15+14,np.minimum(y,273.15+35))

print zone_temperature([288.386, 288.388, 288.39, 288.393], 0.75, 290.353)