FROM pytorchlightning/pytorch_lightning:base-cuda-py3.12-torch2.4-cuda12.4.0
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace
ENV HOME=/workspace

# Copy entire directory to working directory
#COPY . /workspace
# Copy training job code + requirements
COPY requirements.txt /workspace/requirements.txt
# Give access to OVH Cloud User (UID 42420) to necessary directories
# such as /logs, /data or /output directories
RUN chown -R 42420:42420 /workspace
# Install linux packages and dependencies
RUN apt-get update && apt-get install -y git
# Install python dependecies
RUN pip install -r requirements.txt
# Copy config files and code
# (at the end of docker, so that change in scripts do not imply rebuild the entire image...)
COPY experiment.config.yml /workspace/experiment.config.yml
COPY codecarbon.config /workspace/codecarbon.config
COPY ./src /workspace/src


CMD ["python", "-m", "src.main"]