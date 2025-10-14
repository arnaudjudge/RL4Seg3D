FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

COPY . .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y # for cv2
RUN pip install -r docker_requirements.txt && pip install . --no-dependencies && pip install ./patchless-nnUnet/. --no-dependencies && pip install ./vital/. --no-dependencies
ENV PROJECT_ROOT=/workspace/

WORKDIR .
