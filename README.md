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
   
            make build_cpu           ------- for CPU version pytorch
            make build_gpu           ------- for GPU version pytorch
3. check if the docker image is successfully built on your local computer. Type

            docker image ls
    
    If you see a repository with an image name `mpcdrl` from the output, the docker image `mpcdrl` is sucessfully built.

### Test Tutorials
Next step is to test if the docker image can be used for development by running a tutorial example.

1. go to the tutoral folder: `gym-tutorial/test` by typing:

            cd gym-tutorial/test

2. run example script. This is a classic reinforcement control for a cart-pole system. The cart-pole system is built in Modelica and compiled into a jModelica FMU `ModelicaGym_CartPole.fmu`. The learning algorithm `q_learning` is written in `q_learning.py`

   For Linux or MacOS, type

            bash run.sh 

    For Windows OS, type

            run.bat

    The above shell scripts call the docker environment to run the local reinforcement learning experiments as defined in `test_cart_pole_q_learner.py`. The whole run will last around 200 seconds.
