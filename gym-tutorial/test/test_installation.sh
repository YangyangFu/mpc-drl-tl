exec docker run \
 	  --name c_mpcdrl \
      --user=root \
	  --detach=false \
	  -e DISPLAY=${DISPLAY} \
	  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	  --rm \
	  -v `pwd`:/mnt/shared \
	  -i \
      -t \
	  mpcdrl /bin/bash -c "cd /mnt/shared && python /mnt/shared/test_installation.py"
      
exit $
  
