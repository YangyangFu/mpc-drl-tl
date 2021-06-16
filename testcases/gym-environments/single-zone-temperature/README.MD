# Single Zone Temperature Control Environment
This DRL environment is to control the zone cooling temperature setpoint for minimizing energy costs based on a TOU price while maintaining comfortable indoor thermal environment.

  - `v0`: this is a version with discrete action space for a customized DQN trainer 
  
  - `v1`: this is a version with discrete action space for connecting with tianshou DRL library
    - enable user-specific rewards calculation based on energy costs and temperature violations at each time step
    - fix the inconsistency of shapes between observation spaces and states in `v0`
  
  - `v2`: this is a version with continuous action space for connecting with tianshou DRL library
    - the implementaion is the same as `v1`, except that the action space is continuous with user-specified bounds