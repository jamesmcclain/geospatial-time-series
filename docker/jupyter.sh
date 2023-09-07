#!/bin/bash

docker run -p 8888:8888 --shm-size ${2:-16}G --runtime=nvidia -it --rm -v $(pwd):/workdir -v ${1:-/tmp}:/datasets:ro -w /workdir timeseries-jupyter
