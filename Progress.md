
A paper plan for comparing MPC and DRL/TL in building control

- [Objectives](#objectives)
- [Methodology](#methodology)
- [Building System Model](#building-system-model)
- [Progresses](#progresses)
  - [10/26/2020](#10262020)
    - [Next Steps](#next-steps)
  - [10/30/2020](#10302020)
    - [BOPTEST Software Structure](#boptest-software-structure)
- [Meetings](#meetings)
  - [10/28/2020](#10282020)
  - [11/13/2020](#11132020)
  


# Objectives

The purpose here is to investigate how the performance of transfer learning control compared with model predictive control in building energy system.


# Methodology



# Building System Model


# Progresses
## 10/26/2020 
*Deep Reinforcement Learning for Building HVAC Control*: develop a data-driven approach that leverages the deep reinforcement learning (DRL) technique, to intelligently learn the effective strategy for operating the building HVAC systems. We evaluate the performance of our DRL algorithm through simulations using the widely-adopted EnergyPlus tool. Experiments demonstrate that our DRL-based algorithm is more effective in energy cost reduction compared with the traditional rule-based approach, while maintaining the room temperature within desired range.

*One for Many: Transfer Learning for Building HVAC Control*: this paper proposes a tranfer learning control for buildings. The DRL-based control trained from one building is numercially demonstrated to be transfereable to other buildings and locations.

Energyplus is used as a building simulator. The objective of the control is to minimize energy costs while maitaining zone temperature within bounds by controlling the air flow rate in each VAV terminal box.

> $\vec u$: control inputs for building model, here is the flowrate or VAV damper position

> $\vec y$: system response from building model, here is the energy usage.

> $\vec x$: system states from building model, here is the zone temperatures.


For Modelica implementation, we can use existing five-zone system model to predict zone temperature and system-level energy.

***Questions***
1. $\vec u$ in Modelica model should be damper postions. Otherwise, calculating fan power requires a fan power model based on flowrate. Current implementation is based on fan speed and fan pressure head.


***DRL Platform in Modelica***

Here an integrated environment is configured and contained in a docker file. The environment contains:
   - Pytorch: for DRL development as used in NU's papers.
   - jModelica: Modelica-based building model. This part is to provide simulation capability of Modelica models.
   - Opengym: Interface for using Modelica models as DRL environment.

Detailed configuration is in `Dockerfile`.

### Next Steps
- [ ] test VAV.fmu compilation and simulation
- [ ] configure DRL environment using VAV.fmu
- [ ] develop MPC for VAV model

## 10/30/2020

An intermediate idea is to use existing testcases developed from BOPTEST, which contains residential building with single zone and 8 zones, commercial buildings with single zone, 28-zone office, and 32-zone office.

### BOPTEST Software Structure
![1](resources/notes/1-boptest-structure.jpg)

*Emulator Pool* - contains source files of the test cases and temporary files during simulation

*Database*: 

*Simulation Manager*: simulation environment, parses the source files of the emulators.

*HTTP Rest API*: main point of interaction with the BOPTEST platform. Via the HTTP Rest API, the external controller as a client can submit requests for actions such as adding or selecting an emulator to test, extracting information about the emulator, setting simulation settings, starting a simulation, and reading/writing control signal and measurement data.

![2](resources/notes/2-software.jpg)

Emulation model is `wrapper.fmu`.

Simulation manager is `testcase.py`.

HTTP Request API is `restapi.py`.

# Meetings

## 10/28/2020

NU discussed their conceptual design of fault-tolerant deep reinforcement learning control framework.

They proposed an evloving virtual environment to learn system states based on historical virtual data. The predicted value in the virtual environment is compared with measured data from actual environment. If difference is small, have confidence over current measurement. If difference is large, use data from virtual environement.

## 11/13/2020

NU mentioned their needs in the following quoted email:

    Instead of making 5 models for EnergyPlus as we discussed in the morning, could you make one specific-coarse model pair first?

Other information for the models:

    1. control inputs: discrete VAV terminal mass flowrate; for example, 5 level between 0 and maximum flow. This requires revisions of current Modelica VAV terminal model.
    2. control objectives: minimize energy use while maintain room temperature




