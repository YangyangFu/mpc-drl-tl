exec docker run \
      --user=root \
	  --detach=false \
	  -e DISPLAY=${DISPLAY} \
	  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	  --rm \
	  -v `pwd`:/mnt/shared \
	  -i \
      -t \
	  yangyangfu/mpcdrl:cpu_py3 /bin/bash -c "cd /mnt/shared && python /mnt/shared/identify_zone_dynamics_ann.py"  
exit $
