#!/usr/bin/env bash

RESNET=${2:-18}

mkdir -p /data/$1/not-cloudy/
mkdir -p /data/$1/cloudy/
mkdir -p /data/$1/super-cloudy/
mkdir -p /data/$1/super-duper-cloudy/

./train.py --architecture attention-segmenter --resnet-architecture resnet${RESNET} --dataset in-memory-seasonal --series $(cat /data/not-cloudy-list.txt | sed 's,^,/data/BERLIN-SERIES/epsg4326-shifted/,') --target /data/cloudy-labels.tif --train-batches 50 --eval-batches 200 --batch-size 8 --lr 1e-5 1e-5 --num-heads 3 --howmuch 0.33 --resnet-state /data/TRANSFER-MODELS/resnet${RESNET}-best.pth --output-dir /data/$1/not-cloudy/ --size 128

./train.py --architecture attention-segmenter --resnet-architecture resnet${RESNET} --dataset in-memory-seasonal --series $(cat /data/cloudy-list.txt | sed 's,^,/data/BERLIN-SERIES/epsg4326-shifted/,') --target /data/cloudy-labels.tif --train-batches 50 --eval-batches 200 --batch-size 6 --lr 1e-5 1e-5 --num-heads 3 --howmuch 0.33 --output-dir /data/$1/cloudy/ --size 128 --model-state /data/$1/not-cloudy/attention-segmenter-resnet${RESNET}-best.pth

./train.py --architecture attention-segmenter --resnet-architecture resnet${RESNET} --dataset in-memory-seasonal --series $(cat /data/super-cloudy-list.txt | sed 's,^,/data/BERLIN-SERIES/epsg4326-shifted/,') --target /data/cloudy-labels.tif --train-batches 50 --eval-batches 200 --batch-size 8 --lr 1e-5 1e-5 --num-heads 3 --howmuch 0.33 --output-dir /data/$1/super-cloudy/ --size 128 --model-state /data/$1/cloudy/attention-segmenter-resnet${RESNET}-best.pth

./train.py --architecture attention-segmenter --resnet-architecture resnet${RESNET} --dataset in-memory-seasonal --series $(cat /data/super-duper-cloudy-list.txt | sed 's,^,/data/BERLIN-SERIES/epsg4326-shifted/,') --target /data/cloudy-labels.tif --train-batches 50 --eval-batches 200 --batch-size 8 --lr 1e-5 1e-5 --num-heads 3 --howmuch 0.33 --output-dir /data/$1/super-duper-cloudy/ --size 128 --model-state /data/$1/super-cloudy/attention-segmenter-resnet${RESNET}-best.pth

chown 1000:1000 -R /data/$1
