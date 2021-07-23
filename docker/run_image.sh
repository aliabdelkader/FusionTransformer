#!/bin/bash

docker run -it --rm -v /raid/ali/AliThesis/SemanticKitti:/home/user/SemanticKitti -v `pwd`/../../logs:/home/user/logs --network host  --ipc=host --runtime nvidia --gpus '"device=0,1"' fusion_image
