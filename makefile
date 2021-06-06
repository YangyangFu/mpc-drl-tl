IMA_NAME = mpcdrl
IMA_NAME_D = mpcdrl_d

DOCKERFILE_CPU_PY2 = Dockerfile_CPU_PY2
DOCKERFILE_GPU_PY2 = Dockerfile_GPU_PY2
DOCKERFILE_CPU_PY3 = Dockerfile_CPU_PY3
DOCKERFILE_CPU_PY3 = Dockerfile_CPU_PY3

build_cpu_py2:
	docker build -f ${DOCKERFILE_CPU_PY2} --no-cache --rm -t ${IMA_NAME} .

build_gpu_py2:
	docker build -f ${DOCKERFILE_GPU_PY2} --no-cache --rm -t ${IMA_NAME} .

build_cpu_py3:
	docker build -f ${DOCKERFILE_CPU_PY3} --no-cache --rm -t ${IMA_NAME} .

build_gpu_py3:
	docker build -f ${DOCKERFILE_GPU_PY3} --no-cache --rm -t ${IMA_NAME} .

remove_image:
	docker rmi ${IMA_NAME}

push:
	docker push ${IMA_NAME}:latest
