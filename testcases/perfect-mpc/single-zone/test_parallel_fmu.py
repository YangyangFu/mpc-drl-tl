# Import numerical libraries
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# Import the needed JModelica.org Python methods
from pyfmi import load_fmu

from multiprocessing import Pool


def simulate_fmu(u):

    options = fmu.simulate_options()
    options['result_handling'] = "memory"
    fmu.set('uFan',u)
    res = fmu.simulate(201*24*3600., 201*24*3600+3600., options=options)
    return res.final('TRoo')

if __name__ == "__main__":
    fmu = load_fmu("SingleZoneFCU.fmu")
    pool =  Pool(3)
    res = pool.map(simulate_fmu, [0.0,0.5,1.0])
    pool.close()
    pool.join()
    print(res)
