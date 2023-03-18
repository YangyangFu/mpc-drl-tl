# Five Zone Air System Supervisory Control Environment
This DRL environment is to control the setpoints of a five-zone VAV system (only air system) from the supervisory level for minimizing energy consumption while maintaining comfortable indoor thermal environment.

  - `v0`: this is a version with discrete supply air temperature action space (eleven states) for connecting with tianshou DRL library
  - `v1`: this is a version with discrete supply air temperature action space (eight states) for connecting with tianshou DRL library
  - `v2`: this is a version with continuous multi-dimensional action space (eight states, two actions: supply air temperature and dp setpoints) for connecting with tianshou DRL library
