IMA_NAME = mpcdrl
IMA_NAME_D = mpcdrl_debug
HOST = yangyangfu

DOCKERFILE_DEBUG = Dockerfile_debug
DOCKERFILE_CPU_PY3 = Dockerfile_CPU_PY3
DOCKERFILE_GPU_PY3 = Dockerfile_GPU_PY3

TAG_CPU_PY3 = cpu_py3
TAG_GPU_PY3 = gpu_py3

CUDA_VERSION = 11.7
CUDNN_VERSION = 8
TORCH_VERSION = 1.13.1

build_debug:
	docker build -f ${DOCKERFILE_DEBUG} --no-cache --rm -t ${IMA_NAME_D} .

build_cpu_py3:
	docker build -f ${DOCKERFILE_CPU_PY3} --no-cache --rm -t ${IMA_NAME} .

build_gpu_py3:
	docker build -f ${DOCKERFILE_GPU_PY3} --no-cache --rm -t ${IMA_NAME} .

remove_image:
	docker rmi ${IMA_NAME}

tag_cpu_py3:
	docker tag ${IMA_NAME} ${HOST}/${IMA_NAME}:${TAG_CPU_PY3}
tag_gpu_py3:
	docker tag ${IMA_NAME} ${HOST}/${IMA_NAME}:${TAG_GPU_PY3}-torch${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}

push_cpu_py3:
	docker push ${HOST}/${IMA_NAME}:${TAG_CPU_PY3}
push_gpu_py3:
	docker push ${HOST}/${IMA_NAME}:${TAG_GPU_PY3}-torch${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}
