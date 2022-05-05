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
	dmpc:cpu /bin/bash -c \
	"cd /mnt/shared && export PYTHONPATH=\{PYTHONPATH}:/home/developer/github && python /mnt/shared/mpLP_nm_1.py"  
