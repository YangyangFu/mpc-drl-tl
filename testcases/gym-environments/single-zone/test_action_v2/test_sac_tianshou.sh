exec docker run \
      --user=root \
	  --shm-size=4.0gb \
	  --detach=false \
	  -e DISPLAY=${DISPLAY} \
	  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	  --rm \
	  -v `pwd`:/mnt/shared \
	  -i \
      -t \
	  yangyangfu/mpcdrl:cpu_py3 /bin/bash -c \
	  "export TUNE_DISABLE_AUTO_CALLBACK_SYNCER=1 && source activate base && export PYTHONPATH=$PYFMI_PY3_CONDA_PATH:$PYTHONPATH && cd /mnt/shared && python /mnt/shared/test_sac_tianshou.py"  
exit $