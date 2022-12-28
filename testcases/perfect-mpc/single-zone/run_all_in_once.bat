docker run ^
	--user=root ^
	--shm-size=8.0gb ^
	--detach=false ^
	-e DISPLAY=${DISPLAY} ^
	-v /tmp/.X11-unix:/tmp/.X11-unix:rw ^
	--rm ^
	-v %CD%:/mnt/shared ^
	-i ^
	-t ^
	yangyangfu/mpcdrl:gpu_py3-torch1.13.1-cuda11.7-cudnn8 /bin/bash -c ^
	"export TUNE_DISABLE_AUTO_CALLBACK_SYNCER=1 && source activate base && export PYTHONPATH=$PYFMI_PY3_CONDA_PATH:$PYTHONPATH && cd /mnt/shared && python /mnt/shared/run_all_in_once.py"  
