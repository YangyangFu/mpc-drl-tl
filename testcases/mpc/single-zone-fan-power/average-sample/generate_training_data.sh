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
	  mpcdrl /bin/bash -c "cd /mnt/shared && python /mnt/shared/generate_training_data.py"
    exit $