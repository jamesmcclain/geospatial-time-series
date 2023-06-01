#!/bin/bash

docker run --shm-size 48G --runtime=nvidia -it --rm -v $(pwd):/workdir -v ${1:-/tmp}:/datasets:ro -w /workdir timeseries:1.13.1-cuda11.6-cudnn8-devel bash
