FROM nvidia/cuda:12.0.0-devel-ubuntu22.04 

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
    libxext6 \
    wget

RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh -O /tmp/conda.sh \
&& bash /tmp/conda.sh -b -p /opt/conda \
&& rm -rf /tmp/conda.sh

# Set environment variables for Conda
ENV PATH=/opt/conda/bin:$PATH

# Set working directory inside container
WORKDIR /app

# Command to start when container is run (bash shell)
CMD ["/bin/bash"]
