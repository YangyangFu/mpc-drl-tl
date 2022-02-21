import numpy as np

def get_price(time, dt, PH):
    price_tou = [0.02987, 0.02987, 0.02987, 0.02987,
                 0.02987, 0.02987, 0.04667, 0.04667,
                 0.04667, 0.04667, 0.04667, 0.04667,
                 0.15877, 0.15877, 0.15877, 0.15877,
                 0.15877, 0.15877, 0.15877, 0.04667,
                 0.04667, 0.04667, 0.02987, 0.02987]
    t_ph = np.arange(time, time+dt*PH, dt)
    price_ph = [price_tou[int(t % 86400. / 3600)] for t in t_ph]

    return price_ph


time = 17146800
dt = 900. 
PH = 4
price_ph = get_price(time, dt, PH)
print(price_ph)
