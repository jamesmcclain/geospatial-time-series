#!/usr/bin/env bash

mkdir -p /data/$1/not-cloudy/
mkdir -p /data/$1/cloudy/
mkdir -p /data/$1/super-cloudy/
mkdir -p /data/$1/super-duper-cloudy/

./train.py --architecture cheaplab-segmenter --dataset in-memory-seasonal --series $(cat /data/not-cloudy-list.txt | sed 's,^,/data/BERLIN-SERIES/epsg4326-shifted/,') --target /data/cloudy-labels.tif --train-batches 100 --eval-batches 200 --batch-size 12 --lr 1e-3 1e-4 --num-heads 3 --howmuch 0.33 --output-dir /data/$1/not-cloudy/ --size 64 --epochs 13 --phases 1

./train.py --architecture cheaplab-segmenter --dataset in-memory-seasonal --series $(cat /data/super-duper-cloudy-list.txt | sed 's,^,/data/BERLIN-SERIES/epsg4326-shifted/,') --target /data/cloudy-labels.tif --train-batches 100 --eval-batches 200 --batch-size 10 --lr 1e-3 1e-4 --num-heads 3 --howmuch 0.33 --output-dir /data/$1/super-duper-cloudy/ --size 64 --model-state /data/$1/not-cloudy/cheaplab-segmenter-None-best.pth --epochs 13 --phases 1

chown 1000:1000 -R /data/$1
