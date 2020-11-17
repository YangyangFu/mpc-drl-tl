  exec docker run \
 	  --name dc \
          --user=root \
	  --detach=false \
	  -e DISPLAY=${DISPLAY} \
	  -v /tmp/.X11-unix:/tmp/.X11-unix\
	  --rm \
	  -v `pwd`:/mnt/shared \
	  -i \
          -t \
	  fr_dc /bin/bash -c "cd /mnt/shared && python /mnt/shared/simulate_fmu.py"
    exit $
