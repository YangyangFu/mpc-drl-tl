# MPC-DRL-TL
This repo provides a comparison of some advanced control designs for building energy system, especially HVAC system.

- Dockerfile: provides an integrated numerical environment for deploying DRL to a virtual building in Modelica.
- makefile: make command to build local docker image/container.
- gym-tutoral: test the integrated environment using a classic example, that is, controlling of a pole cart.
  

## Installation
The development environment is configured in a virtual Unbuntu OS contained in a docker environment. 

### Requirements
1. Docker: Docker can be downloaded and installed from https://www.docker.com/products/docker-desktop. 

2. Make: Make is a tool to control the generation of executables from the program's source files. On windows, one can download from https://www.cygwin.com/. Make sure the tool is installed in the OS environmental path.

### Installation Steps
After installing the required software, execute the following steps to build and test the docker environment on your local computer.

1. go to folder `MPC-DRL-TL`, and open a terminal. Make sure `Dockerfile_XXX` and `makefile` are in current folder
2. build a local docker image from the provided `Dockerfile_XXX` by typing in the terminal
  
```
          make build_cpu_py3           ------- for CPU version pytorch in Python 3
          make tag_cpu_py3             
          make build_gpu_py3           ------- for GPU version pytorch in Python 3
          make tag_gpu_py3
```
3. check if the docker image is successfully built on your local computer. Type

            docker image ls
    
    If you see a repository with an image name `mpcdrl` from the output, the docker image `mpcdrl` is sucessfully built.

## Testing

### Test Perfect Model Predictive Control

This is to test the perfect MPC which uses the same building model for control as the virtual building model.

1. go to the testcase folders
    ```
    cd testcase/perfect-mpc
    ```
2. run MPC test cases
   
   On Linux or MacOS,
    ```
    bash test_perfect_mpc.sh
    ```

    On Windows OS,
    ```
    test_perfect_mpc.bat
    ```

### Test Model Predictive Control
This is to test the developed model predictive control (MPC) testcases. 

1. go to the testcase folders
    
    ```
    cd testcases/mpc/single-zone
    ```
2. run MPC testcase

    For Linux or MacOS, type
      ```
      bash test_mpc.sh
      ```

    For windows OS, type
      ```
      test_mpc.bat
      ```
 ### Test Deep Reinforcement Learning Control 
This is to test the developed deep reinforcment learning (DRL) control testcases. 

1. go to the testcase folders
    
    ```
    cd mpc-drl-tl/testcases/gym-environments/single-zone/test_action_v1
    ```
2. run DRL testcase

    For Linux or MacOS, type
      ```
      bash test_ddqn_tianshou.sh
      ```
    For windows OS, type
      ```
      test_ddqn_tianshou.bat
      ```
