FROM yangyangfu/jmodelica_py2_gym_pytorch_cpu

USER root

# intall customized gym environment into docker: install a building control environment
## to-be-updated
COPY ./testcases/gym-environments/single-zone/gym_singlezone_jmodelica $HOME/github/testcases/gym-environments/single-zone/gym_singlezone_jmodelica
COPY ./testcases/gym-environments/single-zone/setup.py $HOME/github/testcases/gym-environments/single-zone/setup.py
RUN ls ./github/testcases -l
RUN cd $HOME/github/testcases/gym-environments/single-zone && pip install -e .

WORKDIR $HOME


# copy Modelica dependency to docker container and put them in ModelicaPath
COPY ./library /usr/local/JModelica/ThirdParty/MSL

# Install python package to read epw file
RUN pip install pvlib 
RUN pip install matplotlib

# Add an optimization package ipopt - won't work for python 3 
#ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$IPOPT_HOME
#RUN cd github && git clone https://github.com/mechmotum/cyipopt.git
#RUN cd $HOME/github/cyipopt && ls -l && python setup.py install

# Add optimziation package pyopt
#RUN pip install mpi4py
RUN curl http://www.pyopt.org/_downloads/pyOpt-1.2.0.tar.gz | tar -xz && \
    cd pyOpt-1.2.0 && \
    ls -l &&\
    python setup.py install

# Finish installation
USER developer
# Avoid warning that Matplotlib is building the font cache using fc-list. This may take a moment.
# This needs to be towards the end of the script as the command writes data to
# /home/developer/.cache
RUN python -c "import matplotlib.pyplot"

