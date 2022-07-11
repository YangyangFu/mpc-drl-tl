import json 
import numpy as np

d = np.loadtxt('x_opt.txt')
u_opt = []
for i in d:
    u_opt.append([i])

with open('u0.json', "w") as f:
    json.dump(u_opt, f)