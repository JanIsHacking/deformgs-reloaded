FROM nvidia/cuda:11.7.1-devel-ubuntu22.04 

RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y \
    git python3-dev python3-pip \ 
    libxrender1 \
    libxxf86vm-dev \
    libxfixes-dev \
    libxi-dev \
    libxkbcommon-dev \
    libsm-dev \
    libgl-dev \
    python3-tk \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6