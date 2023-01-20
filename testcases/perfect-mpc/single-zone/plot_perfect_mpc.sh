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
	yangyangfu/mpcdrl:gpu_py3-torch1.13.1-cuda11.7-cudnn8 /bin/bash -c \
	"source activate base && export PYTHONPATH=$PYFMI_PY3_CONDA_PATH:$PYTHONPATH && cd /mnt/shared && python /mnt/shared/plot_perfect_mpc.py --root-dir ./mpc_tuning/100-1-10-new"  
