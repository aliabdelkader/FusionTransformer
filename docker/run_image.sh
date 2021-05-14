#!/bin/bash

docker run -it --rm -v `pwd`/../:/home/user/FusionTransformer -v `pwd`/../../SemanticKitti:/home/user/SemanticKitti -v ~/logs:~/logs --network host  --ipc=host --runtime nvidia --gpus all --name fusion_container fusion_image
