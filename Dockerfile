FROM yangyangfu/jmodelica_py3_gym_pytorch:gpu
LABEL maintainer yangyangfu(yangyang.fu@tamu.edu)

# root 
USER root

### ====================================================================================
### install env for OPC: optimizer, nonlinear system identifier
RUN conda update conda && \
    conda config --add channels conda-forge && \
    conda install casadi scikit-learn matplotlib

#### ====================================================================================
### Install DRL-related
# Install python package to read epw file
RUN pip install pvlib 

#### =====================================================================================
### Install DRL algorithm - tianshou with the latest development in github
#
WORKDIR $HOME
RUN cd $HOME/github && pip install git+https://github.com/thu-ml/tianshou.git@master --upgrade

### Install DRL algorithm - StableBaseline3 in case tianshou is not working - discarded 
#WORKDIR $HOME
#RUN pip install stable-baselines3

### =======================================================================================
### intall customized gym environment into docker: install a building control environment
# install single zone damper control environment
WORKDIR $HOME
COPY ./testcases/gym-environments/single-zone/gym_singlezone_jmodelica $HOME/github/testcases/gym-environments/single-zone/gym_singlezone_jmodelica
COPY ./testcases/gym-environments/single-zone/setup.py $HOME/github/testcases/gym-environments/single-zone/setup.py
RUN cd $HOME/github/testcases/gym-environments/single-zone && pip install -e .

# install single zone temperature control environment
WORKDIR $HOME
COPY ./testcases/gym-environments/single-zone-temperature/gym_singlezone_temperature $HOME/github/testcases/gym-environments/single-zone-temperature/gym_singlezone_temperature
COPY ./testcases/gym-environments/single-zone-temperature/setup.py $HOME/github/testcases/gym-environments/single-zone-temperature/setup.py
RUN cd $HOME/github/testcases/gym-environments/single-zone-temperature && pip install -e .

### =================================================================================
# Install Modelica dependency
COPY ./library /usr/local/JModelica/ThirdParty/MSL

### =========================================================================================
# change user
USER developer
# Avoid warning that Matplotlib is building the font cache using fc-list. This may take a moment.
# This needs to be towards the end of the script as the command writes data to
# /home/developer/.cache
RUN python -c "import matplotlib.pyplot"
