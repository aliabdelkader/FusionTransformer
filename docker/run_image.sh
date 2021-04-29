#!/bin/bash

docker run -it --rm -v `pwd`/../:/home/ubuntu/FusionTransformer -v `pwd`/../../SemanticKitti:/home/ubuntu/SemanticKitti --network host  --ipc=host --runtime nvidia --gpus all --name fusion_container fusion_image
