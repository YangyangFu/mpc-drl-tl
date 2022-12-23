docker run^
      --user=root^
	  --shm-size=8.0gb^
	  --detach=false^
	  -e DISPLAY=${DISPLAY}^
	  -v /tmp/.X11-unix:/tmp/.X11-unix:rw^
	  --rm^
	  -v %CD%:/mnt/shared^
	  -i^
      -t^
	  mpcdrl_debug /bin/bash -c "export TUNE_DISABLE_AUTO_CALLBACK_SYNCER=1 && source activate base && export PYTHONPATH=$PYFMI_PY3_CONDA_PATH:$PYTHONPATH && cd /mnt/shared && python /mnt/shared/test_qrdqn_tianshou.py"  