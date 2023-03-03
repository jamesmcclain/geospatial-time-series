#!/usr/bin/env bash

for data in not-cloudy super-duper-cloudy
do
    for arch in cheaplab-lite-segmenter cheaplab-segmenter
    do
	./view.py --architecture "$arch" --series $(cat /data/"$data"-list.txt | sed 's,^,/data/BERLIN-SERIES/epsg4326-shifted/,') --resnet-a resnet18 --model-state /data/output-$(echo $arch | sed 's,-segmenter,,')/super-duper-cloudy/"$arch"-None-best.pth --size 128 --stride 128 --batch-size 2 --output-dir /data/results/ --device cuda --num-heads 3 --salience --no-prediction --name $(echo $arch | sed 's,-segmenter,,')-"$data"
    done

    for arch in 18 34
    do
	./view.py --architecture attention-segmenter --series $(cat /data/"$data"-list.txt | sed 's,^,/data/BERLIN-SERIES/epsg4326-shifted/,') --resnet-architecture resnet"$arch" --model-state /data/output-resnet"$arch"/super-duper-cloudy/attention-segmenter-resnet"$arch"-best.pth --size 128 --stride 128 --batch-size 2 --output-dir /data/results/ --device cuda --num-heads 3 --salience --no-prediction --name resnet"$arch"-"$data"
    done
done

chown 1000:1000 -R /data/results/
