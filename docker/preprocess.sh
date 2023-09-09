#!/bin/bash

docker run -it --rm -v $(pwd):/workdir -v ${1:-/tmp}:/daylight:ro -w /workdir timeseries-preprocess bash
