#!/bin/bash

docker run -it --rm -v $(pwd):/workdir -w /workdir timeseries-preprocess bash
