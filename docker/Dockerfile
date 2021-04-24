FROM ubuntu:18.04

RUN apt-get -y update
RUN apt-get -y install sudo git gcc g++ libsparsehash-dev apt-utils linux-headers-$(uname -r) libxml2 wget

### install cuda
COPY docker/cuda_10.2.89_440.33.01_linux.run /tmp
RUN /tmp/cuda_10.2.89_440.33.01_linux.run --silent --toolkit --samples
RUN export PATH=/usr/local/cuda/bin:${PATH}
RUN export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Add user ubuntu with no password, add to sudo group
RUN adduser --disabled-password --gecos '' ubuntu
RUN adduser ubuntu sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
WORKDIR /home/ubuntu/
RUN chmod a+rwx /home/ubuntu/

USER ubuntu

# Anaconda installing
RUN wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
RUN bash Anaconda3-5.0.1-Linux-x86_64.sh -b
RUN rm Anaconda3-5.0.1-Linux-x86_64.sh
# Set path to conda
ENV PATH /home/ubuntu/anaconda3/bin:$PATH
RUN conda install anaconda


RUN conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
RUN python -m pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git

COPY ./ /home/ubuntu/FusionTransformer
RUN sudo chown ubuntu /home/ubuntu/FusionTransformer
WORKDIR /home/ubuntu/FusionTransformer
RUN sudo $(which pip) install -e .
RUN pip install timm
RUN pip install dataclasses
