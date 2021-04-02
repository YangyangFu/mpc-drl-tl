IMA_NAME = mpcdrl
DOCKERFILE_CPU = Dockerfile_CPU
DOCKERFILE_GPU = Dockerfile_GPU

build_cpu:
	docker build -f ${DOCKERFILE_CPU} --no-cache --rm -t ${IMA_NAME} .

build_gpu:
	docker build -f ${DOCKERFILE_GPU} --no-cache --rm -t ${IMA_NAME} .

remove_image:
	docker rmi ${IMA_NAME}

push:
	docker push ${IMA_NAME}:latest
