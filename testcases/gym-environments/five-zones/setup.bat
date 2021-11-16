docker run^
      --user=root^
	  -e DISPLAY=${DISPLAY}^
	  -v /tmp/.X11-unix:/tmp/.X11-unix^
	  -v %CD%:/mnt/shared^
	  -dit^
	  mpcdrl /bin/bash -c "source activate base && cd /mnt/shared && pip install -e"

