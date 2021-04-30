FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel
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
    make \
    cmake \
    nano \
    libsparsehash-dev

ENV HOME=/home/user

# All users can use /home/user as their home directory
# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user ${HOME}
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
#USER user

# Create a working directory
RUN chmod 777 ${HOME}
WORKDIR ${HOME}

# Install Miniconda and Python 3.8
#ENV CONDA_AUTO_UPDATE_CONDA=false
#ENV PATH=/home/user/miniconda/bin:$PATH
#RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh \
# && chmod +x ~/miniconda.sh \
# && ~/miniconda.sh -b -p ~/miniconda \
# && rm ~/miniconda.sh \
# && conda install -y python=3.6 \
# && conda clean -ya

#ENV CUDA_HOME=/usr/local/cuda
#RUN conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch && conda clean -ya

#RUN sudo chown user:user -R /opt
RUN git clone https://github.com/mit-han-lab/torchsparse.git \ 
    && cd torchsparse \
    && export CUDA_HOME="/usr/local/cuda" \
    && python setup.py install

COPY ./ ${HOME}/FusionTransformer
#RUN sudo chown user -R ${HOME}/FusionTransformer
WORKDIR ${HOME}/FusionTransformer
RUN python -m pip install --no-cache-dir -e .

#RUN sudo apt-get autoremove -y && sudo apt-get clean && \
 #   sudo rm -rf /var/lib/apt/lists/*

