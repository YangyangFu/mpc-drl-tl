FROM yangyangfu/jmodelica_py3
LABEL maintainer yangyangfu(yangyang.fu@tamu.edu)

# root 
USER root

# update: for pytorch cpu/gpu
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python-opengl 

### ====================================================================================
## install env for OPC: optimizer, nonlinear system identifier
RUN conda update conda && \
    conda config --add channels conda-forge && \
    conda install pip \
         casadi \
         tianshou \
         matplotlib && \
    conda clean -ya

## install fmu-gym
WORKDIR $HOME
RUN pip install git+git://github.com/YangyangFu/modelicagym.git@master 

USER developer
WORKDIR $HOME 
