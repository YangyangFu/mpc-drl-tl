# Simulation Model
## Model description
This model simulate a single-zone DX system. The zone cooling air temperature setpoint can be adjusted by users to test the performance of different controllers.
## Simulation Performance in jModelica

`ncp = 500.`

For a day simulation with an interval of 15 minutes, with writing the results to memory leads to 10.26 seconds. 
Writing results to *.mat file leads to 15.23 seconds.

`npc = 100.`

For a day simulation with an interval of 15 minutes, with writing the results to memory leads to 2.8 seconds.
Writing results to *.mat file leads to 4 seconds.