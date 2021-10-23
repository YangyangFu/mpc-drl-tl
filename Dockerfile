FROM yangyangfu/jmodelica_py3
LABEL maintainer yangyangfu(yangyang.fu@tamu.edu)

# root 
USER root

### ===============================================================================
## install pytorch gpu using cuda
## add cuda
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 10.1.243
ENV CUDA_PKG_VERSION 10-1=$CUDA_VERSION-1

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-$CUDA_PKG_VERSION \
    cuda-compat-10-1 \
    && ln -s cuda-10.1 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.1 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411 brand=tesla,driver>=418,driver<419"

# CUDNN
ENV CUDNN_VERSION 8.0.4.30

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8=$CUDNN_VERSION-1+cuda10.1 \
    libcudnn8-dev=$CUDNN_VERSION-1+cuda10.1 \
    && apt-mark hold libcudnn8 && \
    rm -rf /var/lib/apt/lists/*

# install pyhton opengl - not necessary if no render
#RUN apt-get update && \
#    apt-get install -y --no-install-recommends \
#    python-opengl 

# Install pytorch gpu version
RUN conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch

### ====================================================================================
## install env for OPC: optimizer, nonlinear system identifier
RUN conda update conda && \
    conda config --add channels conda-forge && \
    conda install pip \
    scikit-learn \
    casadi \
    tianshou \
    matplotlib \
    pvlib-python && \
    conda clean -ya

## install fmu-gym
WORKDIR $HOME
RUN pip install git+git://github.com/YangyangFu/modelicagym.git@master 

### =======================================================================================
### intall customized gym environment into docker: install a building control environment
# install single zone damper control environment
WORKDIR $HOME
RUN mkdir github
COPY ./testcases/gym-environments/single-zone/gym_singlezone_jmodelica $HOME/github/testcases/gym-environments/single-zone/gym_singlezone_jmodelica
COPY ./testcases/gym-environments/single-zone/setup.py $HOME/github/testcases/gym-environments/single-zone/setup.py
RUN cd $HOME/github/testcases/gym-environments/single-zone && pip install -e .

# install single zone temperature control environment
WORKDIR $HOME
COPY ./testcases/gym-environments/single-zone-temperature/gym_singlezone_temperature $HOME/github/testcases/gym-environments/single-zone-temperature/gym_singlezone_temperature
COPY ./testcases/gym-environments/single-zone-temperature/setup.py $HOME/github/testcases/gym-environments/single-zone-temperature/setup.py
RUN cd $HOME/github/testcases/gym-environments/single-zone-temperature && pip install -e .

### =============================
USER developer
WORKDIR $HOME 

# Avoid warning that Matplotlib is building the font cache using fc-list. This may take a moment.
# This needs to be towards the end of the script as the command writes data to
# /home/developer/.cache
RUN python -c "import matplotlib.pyplot"