FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive

#---- Install dependencies ----#
RUN apt-get update &&\
    apt-get install -y git python3.11 python3.11-venv python3.11-dev curl wget ffmpeg libsm6 libxext6 &&\
    apt-get clean
# Setup env
# ENV PATH="/venv/bin:$PATH"
# RUN python3.12 -m venv ./venv
# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
# Install python libs
COPY requirements.txt ./requirements.txt
RUN python3.11 -m pip install --upgrade pip
RUN pip install -r requirements.txt
# Download openparse models
RUN openparse-download

#---- Setup workspace ----#
WORKDIR /workspace
ENV HOME=/workspace
# Give access to OVH Cloud User (UID 42420) to necessary directories
# such as /logs, /data or /output directories
RUN chown -R 42420:42420 /workspace
# Sets utf-8 encoding for Python et al
ENV LANG=C.UTF-8
# Turns off writing .pyc files; superfluous on an ephemeral container.
ENV PYTHONDONTWRITEBYTECODE=1
# Seems to speed things up
ENV PYTHONUNBUFFERED=1
# Copy data
COPY ./data /workspace/data
# Copy config files and code
COPY experiment.config.yml /workspace/experiment.config.yml
COPY .codecarbon.config /workspace/.codecarbon.config
# Copy code
COPY ./src /workspace/src


CMD ["python", "-m", "src.main"]