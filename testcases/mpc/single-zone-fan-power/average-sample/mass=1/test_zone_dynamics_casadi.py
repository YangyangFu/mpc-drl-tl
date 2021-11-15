# -*- coding: utf-8 -*-
import numpy as np
import casadi as ca
import model
import joblib

class Zone(ca.Callback):
  def __init__(self, name, Lz=4, Lo=4, json_file="", opts={"enable_fd":True}):
    ca.Callback.__init__(self)
    self.Lz=Lz,
    self.Lo=Lo,
    self.json_file = json_file
    self.construct(name, opts)

  # Number of inputs and outputs
  def get_n_in(self): return 1
  def get_n_out(self): return 1

  # Array of inputs and outputs
  def get_sparsity_in(self,i):
    #self.nin=2*np.asarray(self.Lz,dtype="int32")+np.asarray(self.Lo,dtype="int32")+2
    self.nin=2*self.Lz[0]+self.Lo[0]+2
    return ca.Sparsity.dense(self.nin,1)
  def get_sparsity_out(self,i):
    return ca.Sparsity.dense(1,1)

  # Initialize the object
  def init(self):
     print('initializing object')

  # Evaluate numerically
  def eval(self, arg):
    x = arg[0] # size of sparsity_in
    T = model.Zone(Lz=self.Lz[0], Lo=self.Lo[0], json_file=self.json_file)

    Tz_his_meas=x[0:self.Lz[0]]
    Tz_his_pred=x[self.Lz[0]:self.Lz[0]+self.Lz[0]]
    To_his_meas=x[self.Lz[0]+self.Lz[0]:self.Lz[0]+self.Lz[0]+self.Lo[0]]
    mz=x[-2]
    Tsa=x[-1]
    f = T.predict(Tz_his_meas, Tz_his_pred, To_his_meas, mz, Tsa)
    #Tz_his_meas, Tz_his_pred, To_his_meas, mz, Tsa
    return [f]

class Zone1(ca.Callback):
  def __init__(self, name, ann_file="", opts={"enable_fd":True}):
    ca.Callback.__init__(self)
    self.ann = joblib.load(ann_file)
    self.construct(name, opts)

  # Number of inputs and outputs
  def get_n_in(self): return 1
  def get_n_out(self): return 1

  # Array of inputs and outputs
  def get_sparsity_in(self,i):
    #self.nin=2*np.asarray(self.Lz,dtype="int32")+np.asarray(self.Lo,dtype="int32")+2
    return ca.Sparsity.dense(9,1)
  def get_sparsity_out(self,i):
    return ca.Sparsity.dense(1,1)

  # Initialize the object
  def init(self):
     print('initializing object')

  # Evaluate numerically
  def eval(self, arg):
    x = arg[0] # size of sparsity_in
    f = self.ann.predict(np.transpose(np.array(x)))
    #Tz_his_meas, Tz_his_pred, To_his_meas, mz, Tsa
    return f

## formulate optimization
T = Zone(name='Tz', Lz=4, Lo=4, json_file='zone_arx.json')
x = ca.MX.sym("x",14)
f=T(x)
#x=[22, 22., 23, 24, 22, 22., 23, 24, 30, 30, 30, 30, 0.7, 13]
#print(T(x))

## formulate optimization
T1 = Zone1(name='Tz', ann_file='zone_ann.pkl')
x1 = ca.MX.sym("x1",9)
f1 = T1(x1)
#x1=[22, 22., 23, 24, 30, 30, 30, 30, 0.7]
#print(T1(x1))

## Optimize arx 
nlp = {'x':x, 'f':f}
solver = ca.nlpsol("nlpsol", "ipopt", nlp)
# solve NLP
sol = solver(lbx=[22, 22., 23, 24, 22, 22., 23, 24, 30, 30, 30, 30, 0., 13], ubx=[22, 22., 23, 24, 22, 22., 23, 24, 30, 30, 30, 30, 0.7, 13])
# Print solution
print("-----")
print("objective at solution = ", sol["f"])
print("primal solution = ", sol["x"])
print("dual solution (x) = ", sol["lam_x"])
print("dual solution (g) = ", sol["lam_g"])

## Optimize ann
nlp = {'x':x1, 'f':f1}
solver = ca.nlpsol("nlpsol", "ipopt", nlp)
# solve NLP
sol = solver(lbx=[22, 22., 23, 24, 30, 30, 30, 30, 0.], ubx=[22, 22., 23, 24, 30, 30, 30, 30, 0.7])
# Print solution
print("-----")
print("objective at solution = ", sol["f"])
print("primal solution = ", sol["x"])
print("dual solution (x) = ", sol["lam_x"])
print("dual solution (g) = ", sol["lam_g"])