docker run^
      --user=root^
	  --detach=false^
	  -e DISPLAY=${DISPLAY}^
	  -v /tmp/.X11-unix:/tmp/.X11-unix^
	  --rm^
	  -v %CD%:/mnt/shared^
	  -i^
      -t^
	  yangyangfu/mpcdrl:cpu_py2 /bin/bash -c "cd /mnt/shared && python /mnt/shared/mpc_final_run.py"

