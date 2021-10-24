# -*- coding: utf-8 -*-
import numpy as np
from casadi import *

# Declare variables
x = SX.sym("x",2)

# Form the NLP
def obj(x):
  return x[0]**2 + x[1]**2 # objective

def con(x):
  return x[0]+x[1]-10      # constraint
f=obj(x)
g=con(x)

nlp = {'x':x, 'f':f, 'g':g}

# Pick an NLP solver
MySolver = "ipopt"
#MySolver = "worhp"
#MySolver = "sqpmethod"

# Solver options
opts = {}
if MySolver=="sqpmethod":
  opts["qpsol"] = "qpoases"
  opts["qpsol_options"] = {"printLevel":"none"}

# Allocate a solver
solver = nlpsol("solver", MySolver, nlp, opts)

# Solve the NLP
sol = solver(lbg=0)

# Print solution
print("-----")
print("objective at solution = ", sol["f"])
print("primal solution = ", sol["x"])
print("dual solution (x) = ", sol["lam_x"])
print("dual solution (g) = ", sol["lam_g"])

