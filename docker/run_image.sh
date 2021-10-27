#!/bin/bash


docker run -it --rm -v /home/mrafaat/SemanticKitti:/home/user/SemanticKitti -v /home/mrafaat/AliThesis/logs:/home/user/logs --network host  --ipc=host --runtime nvidia --gpus '"device=0,1"' fusion_image

