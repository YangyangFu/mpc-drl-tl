FROM yangyangfu/gym_torch_jmodelica_py2

# install modelicagym from source
USER root
WORKDIR $HOME
RUN echo $HOME
RUN mkdir github && cd github && git clone https://github.com/YangyangFu/modelicagym.git && \
    cd modelicagym && python -m pip install -e .

WORKDIR $HOME

# intall customized gym environment into docker: this is a tutorial: pole cart tutorial 
COPY ./gym-tutorial/gym_cart_jmodelica $HOME/github/Tutorials/gym_cart_jmodelica
COPY ./gym-tutorial/setup.py $HOME/github/Tutorials/setup.py
RUN ls ./github/Tutorials -l
RUN cd $HOME/github/Tutorials && pip install -e .

WORKDIR $HOME

# intall customized gym environment into docker: install a building control environment
## to-be-updated



# copy Modelica dependency to docker container and put them in ModelicaPath
COPY ./library /usr/local/JModelica/ThirdParty/MSL
WORKDIR $HOME