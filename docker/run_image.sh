#!/bin/bash

docker run -it --rm -v `pwd`/../:/home/ubuntu/FusionTransformer --network host --runtime nvidia --gpus all --name fusion_container fusion_image
