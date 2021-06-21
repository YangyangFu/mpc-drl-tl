# Single Zone Fan Speed Control Environment
This DRL environment is to control the fan speed of a fan-coil unit system for minimizing energy costs based on a TOU price while maintaining comfortable indoor thermal environment.

  - `v0`: this is a version for a customized DQN trainer.
  
  - `v1`: this is a version for connecting with tianshou DRL library
    - enable user-specific rewards calculation based on energy costs and temperature violations at each time step
    - fix the inconsistency of shapes between observation spaces and states in `v0`  

  - `v1.1`: this version uses future energy prices as additional DRL states compared with `v1`.
  
  - `v2`: this is a version with continuous action space for connecting with tianshou DRL library
    - the implementaion is the same as `v1`, except that the action space is continuous with user-specified bounds