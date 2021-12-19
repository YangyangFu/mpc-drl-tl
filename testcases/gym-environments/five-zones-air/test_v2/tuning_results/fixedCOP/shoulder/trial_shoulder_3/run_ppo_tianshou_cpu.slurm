#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=test_ppo_v1_air       #Set the job name to "JobExample5"
#SBATCH --time=1-8:00:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --mem=5G                  #Request 10GB per node
#SBATCH --output=log_ppo_v1_air.%j      #Send stdout/err to "Example5Out.[jobID]"

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=122782406486             #Set billing account
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=xing.lu@tamu.edu    #Send all emails to email_address 

ml purge
# ADD udocker to path before loading
echo "Define udocker path!!!!"
export UDOCKER_DIR=$SCRATCH/udocker/
export PATH=$UDOCKER_DIR:$PATH
#ml udocker
# define a container name
export CONTAINER_ID=ppo_v1_air
# create a container
echo "creat a DRL container!!!!!!"
# udocker create --name=$CONTAINER_ID solutionlx129/drlair:cpu
# run experiment
echo "run DRL experiment in Docker"
udocker run --user=root -e DISPLAY=${DISPLAY} -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v `pwd`:/mnt/shared $CONTAINER_ID /bin/bash -c "source activate base && export PYTHONPATH=$PYFMI_PY3_CONDA_PATH:$PYTHONPATH && cd /mnt/shared && python /mnt/shared/test_ppo_tianshou.py"
# remove container after experiment
# udocker rm $CONTAINER_ID
#bash test_dqn_tianshou_udocker.sh
