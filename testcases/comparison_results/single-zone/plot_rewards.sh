exec docker run \
      --user=root \
	  --detach=false \
	  -e DISPLAY=${DISPLAY} \
	  -v /tmp/.X11-unix:/tmp/.X11-unix \
	  --rm \
	  -v `pwd`:/mnt/shared \
	  -i \
      -t \
	  mpcdrl_debug /bin/bash -c "cd /mnt/shared && python /mnt/shared/plot_rewards.py --root-dir ./v2_action"
exit $