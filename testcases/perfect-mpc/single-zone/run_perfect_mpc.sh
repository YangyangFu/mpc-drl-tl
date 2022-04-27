docker run \
	--user=root \
	--shm-size=4.0gb \
	--detach=false \
	-e DISPLAY=${DISPLAY} \
	-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	--rm \
	-v `pwd`:/mnt/shared \
	-i \
	-t \
	mpcdrl_perfect_mpc /bin/bash -c \
	"cd /mnt/shared && python /mnt/shared/run_perfect_mpc.py"  
