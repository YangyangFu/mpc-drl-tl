IMA_NAME = mpcdrl
IMA_NAME_D = mpcdrl_d

DOCKERFILE_CPU = Dockerfile_CPU
DOCKERFILE_GPU = Dockerfile_GPU

DOCKERFILE_CPU_D = Dockerfile_CPU_D

build_cpu:
	docker build -f ${DOCKERFILE_CPU} --no-cache --rm -t ${IMA_NAME} .

build_gpu:
	docker build -f ${DOCKERFILE_GPU} --no-cache --rm -t ${IMA_NAME} .

build_cpu_debug:
	docker build -f ${DOCKERFILE_CPU_D} --no-cache --rm -t ${IMA_NAME_D} .

remove_image:
	docker rmi ${IMA_NAME}

push:
	docker push ${IMA_NAME}:latest
