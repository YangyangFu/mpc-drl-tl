docker run --name fmuc1^
      --user=root^
	  --detach=false^
	  -e DISPLAY=${DISPLAY}^
	  -v /tmp/.X11-unix:/tmp/.X11-unix^
	  --rm^
	  -v %CD%:/mnt/shared^
	  -i^
      -t^
	  mpcdrl /bin/bash -c "cd /mnt/shared && python /mnt/shared/test_jmodelica_ipopt.py"

