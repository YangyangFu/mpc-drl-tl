  exec docker run \
 	  --name test \
          --user=root \
	  --detach=false \
	  -e DISPLAY=${DISPLAY} \
	  -v /tmp/.X11-unix:/tmp/.X11-unix\
	  --rm \
	  -v `pwd`:/mnt/shared \
	  -i \
          -t \
	  fncs /bin/bash -c "cd /mnt/shared && python /mnt/shared/simulate_fmu.py"
    exit $
