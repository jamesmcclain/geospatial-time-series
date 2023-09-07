#!/bin/bash

docker run --shm-size ${2:-48}G --runtime=nvidia -it --rm -v $(pwd):/workdir -v ${1:-/tmp}:/datasets:ro -w /workdir timeseries:2.0.1-cuda11.7-cudnn8-devel bash
