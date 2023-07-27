#!/bin/bash

docker run -p 8888:8888 --shm-size 16G --runtime=nvidia -it --rm -v $(pwd):/workdir -v ${1:-/tmp}:/datasets:ro -w /workdir timeseries:2.0.1-cuda11.7-cudnn8-jupyter
