import pandas as pd 
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

res = pd.read_csv('results_opt.csv', index_col=[0])

plt.figure()
plt.plot(res.index, res['TRoo']-273.15)
plt.ylabel('Temperature [C]')
plt.savefig('results_opt.png')