FROM yangyangfu/jmodelica_py2_gym
LABEL maintainer yangyangfu(yangyang.fu@tamu.edu)

## add cuda
USER root
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
# Install pytorch=1.4.0, which is the lastest version that supports python 2
# add cuda 10.1 supported pytorch, and add --no-cache-dir to avoid large memory use
RUN pip install --default-timeout=100 torch==1.4.0 torchvision==0.5.0  --no-cache-dir

### intall customized gym environment into docker: install a building control environment
# to-be-updated
WORKDIR $HOME
COPY ./testcases/gym-environments/single-zone/gym_singlezone_jmodelica $HOME/github/testcases/gym-environments/single-zone/gym_singlezone_jmodelica
COPY ./testcases/gym-environments/single-zone/setup.py $HOME/github/testcases/gym-environments/single-zone/setup.py
RUN ls ./github/testcases -l
RUN cd $HOME/github/testcases/gym-environments/single-zone && pip install -e .

# copy Modelica dependency to docker container and put them in ModelicaPath
WORKDIR $HOME
COPY ./library /usr/local/JModelica/ThirdParty/MSL

# Install python package to read epw file
RUN pip install pvlib 

### Add an optimization package openopt - a wrapper for ipopt
# Get Install Ipopt and delete source code after installation
# IPOPT have different folder structures after 3.13. 
RUN mkdir /usr/local/src
ENV SRC_DIR /usr/local/src
ENV IPOPT_3_12_13_HOME /usr/local/Ipopt-3.12.13

RUN cd $SRC_DIR && \
    wget wget -O - http://www.coin-or.org/download/source/Ipopt/Ipopt-3.12.13.tgz | tar xzf - && \
    cd $SRC_DIR/Ipopt-3.12.13/ThirdParty/Blas && \
    ./get.Blas && \
    cd $SRC_DIR/Ipopt-3.12.13/ThirdParty/Lapack && \
    ./get.Lapack && \
    cd $SRC_DIR/Ipopt-3.12.13/ThirdParty/ASL && \
    ./get.ASL && \    
    cd $SRC_DIR/Ipopt-3.12.13/ThirdParty/Mumps && \
    ./get.Mumps && \
    cd $SRC_DIR/Ipopt-3.12.13/ThirdParty/Metis && \
    ./get.Metis && \
    mkdir $SRC_DIR/Ipopt-3.12.13/build && \
    cd $SRC_DIR/Ipopt-3.12.13/build && \
    cd $SRC_DIR/Ipopt-3.12.13 && \
    ./configure --prefix=$IPOPT_3_12_13_HOME && \
    make &&\
    make install && \
    rm -rf $SRC_DIR

# specify link file location in linux
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$IPOPT_3_12_13_HOME:$IPOPT_3_12_13_HOME/lib:$IPOPT_3_12_13_HOME/bin
ENV PATH $PATH:$LD_LIBRARY_PATH

# install openopt
RUN pip install openopt
RUN pip install FuncDesigner
RUN pip install DerApproximator

# connect ipopt with openopt
# install ipopt python wrapper
RUN cd $HOME/github && git clone https://github.com/YangyangFu/pyipopt.git && ls -l
RUN cd $HOME/github/pyipopt && python setup.py build && python setup.py install

### Finish installation
# add plotting library
RUN pip install matplotlib
# Install scikit learn
RUN pip install -U scikit-learn

# change user
USER developer
# Avoid warning that Matplotlib is building the font cache using fc-list. This may take a moment.
# This needs to be towards the end of the script as the command writes data to
# /home/developer/.cache
RUN python -c "import matplotlib.pyplot"
