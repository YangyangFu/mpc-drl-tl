# Single Zone Fan Speed Control Environment
This DRL environment is to control the fan speed of a fan-coil unit system for minimizing energy costs based on a TOU price while maintaining comfortable indoor thermal environment.

For discrete DRL algorithms:

  - `JModelicaCSSingleZoneEnv-v0`: this is a version for a customized DQN trainer. This will be removed in the future.
  
  - `JModelicaCSSingleZoneEnv-v1`: this is a basic version for connecting with tianshou DRL library. The state and reward designs can be referred to `./gym_singlezone_jmodelica/envs/single_zone_env_v1.py`.
 
  - `JModelicaCSSingleZoneEnv-price-v1`: this version uses future energy prices as additional DRL states compared with `JModelicaCSSingleZoneEnv-v1` as implemented in `./gym_singlezone_jmodelica/envs/single_zone_price_env_v1.py`.
  
  - `JModelicaCSSingleZoneEnv-action-v1`: this version extends `JModelicaCSSingleZoneEnv-price-v1` by adding control action slew rate as part of the reward design, which is implemented in `./gym_singlezone_jmodelica/envs/single_zone_action_env_v1.py`.

For continous DRL algorithms: the corresponding environment are labelled with `v2`.