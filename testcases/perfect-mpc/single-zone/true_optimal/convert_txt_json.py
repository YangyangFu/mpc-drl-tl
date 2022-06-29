import json
import numpy as np
data = np.loadtxt("x_opt.txt")
n = len(data)

u_opt = []
t_opt = []
for i in range(n):
    u_opt.append([data[i]])
    t_opt.append([i])

d = {"u_opt": u_opt,
    "t_opt": t_opt}

with open("u_opt.json", 'w') as f:
    json.dump(d, f)