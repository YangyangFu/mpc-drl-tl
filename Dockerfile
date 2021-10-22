FROM yangyangfu/jmodelica_py3
LABEL maintainer yangyangfu(yangyang.fu@tamu.edu)

# root 
USER root

### ====================================================================================
### install env for OPC: optimizer, nonlinear system identifier
RUN conda update conda && \
    conda config --add channels conda-forge && \
    conda install casadi scikit-learn matplotlib

USER developer
WORKDIR $HOME 
