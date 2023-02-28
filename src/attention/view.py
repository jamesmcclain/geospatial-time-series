#!/usr/bin/env python3

import argparse
import copy
import logging
import sys

import numpy as np
import rasterio as rio
import torch
import tqdm
from rasterio.windows import Window

from models import AttentionSegmenter, CheaplabSegmenter

ARCHITECTURES = ['attention-segmenter', 'cheaplab-segmenter']
DATASETS = ['in-memory-seasonal']
RESNETS = ['resnet18', 'resnet34']


def cli_parser():
    # yapf: disable
    parser = argparse.ArgumentParser()

    parser.add_argument('--architecture', required=True, type=str, choices=ARCHITECTURES)
    parser.add_argument('--batch-size', required=False, type=int, default=32)
    parser.add_argument('--device', required=True, type=str, choices=['cuda', 'cpu'])
    parser.add_argument('--model-state', required=True, type=str)
    parser.add_argument('--name', required=False, type=str, default=None)
    parser.add_argument('--num-heads', required=False, type=int, default=3)
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--resnet-architecture', required=False, type=str, choices=RESNETS)
    parser.add_argument('--series', required=True, type=str, nargs='+')
    parser.add_argument('--size', required=False, type=int, default=256)
    parser.add_argument('--stride', required=False, type=int, default=107)

    return parser
    # yapf: enable


if __name__ == '__main__':
    args = cli_parser().parse_args()
    logging.basicConfig(stream=sys.stderr,
                        level=logging.INFO,
                        format='%(asctime)-15s %(message)s')
    log = logging.getLogger()

    class_names = ["other", "farm", "forest", "road"]
    log.info(args.__dict__)

    device = torch.device(args.device)

    if args.architecture == 'attention-segmenter':
        assert args.resnet_architecture is not None
        model = AttentionSegmenter(
            args.resnet_architecture,
            None,
            args.size,
            num_heads=args.num_heads,
        )
    elif args.architecture == 'cheaplab-segmenter':
        model = CheaplabSegmenter(num_heads=args.num_heads, )
    else:
        pass

    if args.model_state is not None:
        log.info(f'Loading state from {args.model_state}')
        model.load_state_dict(torch.load(args.model_state,
                                         map_location=torch.device('cpu')),
                              strict=True)

    model = model.to(device)
    model.eval()

    raw_profile = None
    out_profile = None
    if args.name is not None:
        raw_outfile = f'{args.output_dir}/{args.name}-raw.tif'
        out_outfile = f'{args.output_dir}/{args.name}-out.tif'
    else:
        raw_outfile = f'{args.output_dir}/raw.tif'
        out_outfile = f'{args.output_dir}/out.tif'

    # Open r/o datasets
    in_datasets = []
    for infile in args.series:
        in_datasets.append(rio.open(infile, 'r'))
        if out_profile is None or raw_profile is None:
            width = in_datasets[-1].width
            height = in_datasets[-1].height

            raw_profile = copy.deepcopy(in_datasets[-1].profile)
            raw_profile.update({
                'compress': 'lzw',
                'predictor': 2,
                'dtype': np.float32,
                'count': 4,
                'bigtiff': 'yes',
                'sparse_ok': True,
                'tiled': True,
            })
            del raw_profile['nodata']
            raw_data = torch.zeros((4, height, width),
                                   dtype=torch.float32).to(device)

            out_profile = copy.deepcopy(in_datasets[-1].profile)
            out_profile.update({
                'compress': 'lzw',
                'predictor': 2,
                'dtype': np.uint8,
                'count': 1,
                'bigtiff': 'yes',
                'sparse_ok': True,
                'tiled': True,
            })
            del out_profile['nodata']
            out_data = np.zeros((1, height, width), dtype=np.float32)

    # Generate list of windows
    windows = []
    for i in range(0, width, args.stride):
        for j in range(0, height, args.stride):
            if i + args.size > width:
                i = width - args.size
            if j + args.size > height:
                j = height - args.size
            windows.append(Window(i, j, args.size, args.size))

    # Gather windows into batches
    batches = [
        windows[i:i + args.batch_size]
        for i in range(0, len(windows), args.batch_size)
    ]

    # Inference
    with torch.no_grad():
        for batch in tqdm.tqdm(batches):
            batch_stack = np.stack([
                np.stack([ds.read(window=window) for ds in in_datasets])
                for window in batch
            ]).astype(np.float32)
            batch_stack = torch.from_numpy(batch_stack).to(dtype=torch.float32,
                                                           device=device)
            raw = model(batch_stack)
            for i, window in enumerate(batch):
                x = window.col_off
                y = window.row_off
                w = window.width
                h = window.height
                raw_data[:, y:(y + h), x:(x + w)] += raw[i, :, :, :].detach()

    raw_data = raw_data.softmax(dim=0).cpu().numpy()
    out_data = np.expand_dims(np.argmax(raw_data, axis=0), axis=0)

    # Close r/o datasets
    for dataset in in_datasets:
        dataset.close()

    # Output results
    log.info(f'Writing {raw_outfile} to disk')
    with rio.open(raw_outfile, 'w', **raw_profile) as ds:
        ds.write(raw_data)
    log.info(f'Writing {out_outfile} to disk')
    with rio.open(out_outfile, 'w', **out_profile) as ds:
        ds.write(out_data)
        ds.write_colormap(
            1, {
                0: (0xff, 0, 0, 0xff),
                1: (0, 0xff, 0),
                2: (0, 0, 0xff),
                3: (0x16, 0x16, 0x1d)
            })
