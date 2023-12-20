docker run^
      --user=root^
	  --shm-size=6.03gb^
	  --detach=false^
	  -e DISPLAY=${DISPLAY}^
	  -v /tmp/.X11-unix:/tmp/.X11-unix:rw^
	  --rm^
	  -v %CD%:/mnt/shared^
	  -i^
      -t^
	  cpu_py3_image /bin/bash -c "source activate base && export PYTHONPATH=$PYFMI_PY3_CONDA_PATH:$PYTHONPATH && cd /mnt/shared && python /mnt/shared/test_gail_tianshou.py"  