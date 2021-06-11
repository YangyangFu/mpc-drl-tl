# Single Zone Temperature Control Environment
This DRL environment is to control the zone cooling temperature setpoint for minimizing energy costs based on a TOU price while maintaining comfortable indoor thermal environment.

  - `v0`: this is a version for a customized DQN trainer.
  
  - `v1`: this is a version for connecting with tianshou DRL library
    - enable user-specific rewards calculation based on energy costs and temperature violations at each time step
    - fix the inconsistency of shapes between observation spaces and states in `v0`