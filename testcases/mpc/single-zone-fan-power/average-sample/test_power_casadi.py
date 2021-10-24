# -*- coding: utf-8 -*-
import casadi as ca
from model import FanPower

class Power(ca.Callback):
  def __init__(self, name, json_file, opts={"enable_fd":True}):
    ca.Callback.__init__(self)
    self.json_file = json_file
    self.construct(name, opts)

  # Number of inputs and outputs
  def get_n_in(self): return 1
  def get_n_out(self): return 1

  # Initialize the object
  def init(self):
     print('initializing object')

  # Evaluate numerically
  def eval(self, arg):
    x = arg[0]
    fan = FanPower(n=4,json_file=self.json_file)
    f = fan.predict(x)
    return [f]

## Test single call
p = Power('p', 'power.json')
x=[0.5]
print(p(x))

## formulate optimization
x = ca.MX.sym("x")
f = p(x)
nlp = {'x':x, 'f':f}
solver = ca.nlpsol("nlpsol", "ipopt", nlp)
# solve NLP
sol = solver(lbx=0.1, ubx=0.8)
# Print solution
print("-----")
print("objective at solution = ", sol["f"])
print("primal solution = ", sol["x"])
print("dual solution (x) = ", sol["lam_x"])
print("dual solution (g) = ", sol["lam_g"])