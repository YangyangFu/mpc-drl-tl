docker run^
      --user=root^
	  --detach=false^
	  -e DISPLAY=${DISPLAY}^
	  -v /tmp/.X11-unix:/tmp/.X11-unix^
	  --rm^
	  -v %CD%:/mnt/shared^
	  -i^
      -t^
	  mpcdrl /bin/bash -c^
	  "source activate base && export PYTHONPATH=$PYFMI_PY3_CONDA_PATH:$PYTHONPATH && cd /mnt/shared && python /mnt/shared/simulate_fmu_baseline.py"
