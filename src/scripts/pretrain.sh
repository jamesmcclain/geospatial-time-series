#!/usr/bin/bash

python3 ./src/pretrain/train.py --input-dir /data/ --output-dir ./ --epochs 42 --batch 32 --architecture cheaplab --train 1024 --eval 128
python3 ./src/pretrain/train.py --input-dir /data/ --output-dir ./ --epochs 42 --batch 8 --architecture resnet50
python3 ./src/pretrain/train.py --input-dir /data/ --output-dir ./ --epochs 42 --batch 12 --architecture resnet34
python3 ./src/pretrain/train.py --input-dir /data/ --output-dir ./ --epochs 42 --batch 16 --architecture resnet18
