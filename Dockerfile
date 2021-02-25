FROM yangyangfu/jmodelica_py2_gym_pytorch_cpu

USER root

### intall customized gym environment into docker: install a building control environment
# to-be-updated
COPY ./testcases/gym-environments/single-zone/gym_singlezone_jmodelica $HOME/github/testcases/gym-environments/single-zone/gym_singlezone_jmodelica
COPY ./testcases/gym-environments/single-zone/setup.py $HOME/github/testcases/gym-environments/single-zone/setup.py
RUN ls ./github/testcases -l
RUN cd $HOME/github/testcases/gym-environments/single-zone && pip install -e .

WORKDIR $HOME

# copy Modelica dependency to docker container and put them in ModelicaPath
COPY ./library /usr/local/JModelica/ThirdParty/MSL

# Install python package to read epw file
RUN pip install pvlib 

### Add an optimization package pyomo - a wrapper for ipopt
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

### Finish installation
# add plotting library
RUN pip install matplotlib
# change user
USER developer
# Avoid warning that Matplotlib is building the font cache using fc-list. This may take a moment.
# This needs to be towards the end of the script as the command writes data to
# /home/developer/.cache
RUN python -c "import matplotlib.pyplot"
