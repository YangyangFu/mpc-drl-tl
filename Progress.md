
A paper plan for comparing MPC and DRL/TL in building control

- [Objectives](#objectives)
- [Methodology](#methodology)
- [Building System Model](#building-system-model)
- [Progresses](#progresses)
  - [10/26/2020](#10262020)
    - [Next Steps](#next-steps)
- [Meetings](#meetings)
  - [10/28/2020](#10282020)
  


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



# Meetings

## 10/28/2020

