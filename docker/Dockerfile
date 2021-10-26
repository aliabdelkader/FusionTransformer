FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    gcc \
    g++ \
    gdb \
    make \
    cmake \
    nano \
    libsparsehash-dev \
    wget \
    unzip \
    openmpi-bin \
    libopenmpi-dev

RUN chmod -R  a+rwX /opt/conda
RUN useradd -ms /bin/bash user
#RUN useradd -rm -d /home/user -s /bin/bash -g root -G sudo -u 1001 user
USER user    
ENV HOME=/home/user
ENV PATH="${PATH}:/user/.local/bin"
# Create a working directory
WORKDIR ${HOME}
#RUN chmod 777 ${HOME}

RUN wget https://github.com/mit-han-lab/torchsparse/archive/refs/tags/v1.1.0.zip && unzip v1.1.0.zip
 
COPY --chown=user:user ./ ${HOME}/FusionTransformer
WORKDIR ${HOME}/FusionTransformer
RUN python -m pip install --no-cache-dir -e .

