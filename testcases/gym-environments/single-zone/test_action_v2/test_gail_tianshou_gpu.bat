docker run^
      --user=root^
	  --gpus all^
	  --shm-size=6.03gb^
	  --detach=false^
	  -e DISPLAY=${DISPLAY}^
	  -v /tmp/.X11-unix:/tmp/.X11-unix:rw^
	  --rm^
	  -v %CD%:/mnt/shared^
	  -i^
      -t^
	  mingzhe37/mpc-drl-gpu:v0 /bin/bash -c "source activate base && export PYTHONPATH=$PYFMI_PY3_CONDA_PATH:$PYTHONPATH && cd /mnt/shared && python /mnt/shared/test_gail_tianshou_gpu.py"  