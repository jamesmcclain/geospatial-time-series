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

from models import AttentionSegmenter, CheaplabLiteSegmenter, CheaplabSegmenter

ARCHITECTURES = [
    'attention-segmenter', 'cheaplab-segmenter', 'cheaplab-lite-segmenter'
]
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
    parser.add_argument('--no-prediction', dest='prediction', default=True, action='store_false')
    parser.add_argument('--num-heads', required=False, type=int, default=3)
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--preshrink', required=False, type=int, default=1)
    parser.add_argument('--resnet-architecture', required=False, type=str, choices=RESNETS)
    parser.add_argument('--salience', dest='salience', default=False, action='store_true')
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
        model = CheaplabSegmenter(num_heads=args.num_heads,
                                  preshrink=args.preshrink)
    elif args.architecture == 'cheaplab-lite-segmenter':
        model = CheaplabLiteSegmenter(num_heads=args.num_heads,
                                  preshrink=args.preshrink)
    else:
        pass

    log.info(f'Loading state from {args.model_state}')
    model.load_state_dict(torch.load(args.model_state,
                                     map_location=torch.device('cpu')),
                          strict=True)

    model = model.to(device)
    model.eval()

    if args.salience:
        assert args.size % args.stride == 0

    raw_profile = None
    out_profile = None
    sal_profile = None
    if args.name is not None:
        raw_outfile = f'{args.output_dir}/{args.name}-raw.tiff'
        out_outfile = f'{args.output_dir}/{args.name}-out.tiff'
        sal_outfile = f'{args.output_dir}/{args.name}-sal.tiff'
    else:
        raw_outfile = f'{args.output_dir}/raw.tiff'
        out_outfile = f'{args.output_dir}/out.tiff'
        sal_outfile = f'{args.output_dir}/sal.tiff'

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

            sal_profile = copy.deepcopy(in_datasets[-1].profile)
            sal_profile.update({
                'compress': 'lzw',
                'predictor': 2,
                'dtype': np.float32,
                'count': 3,
                'bigtiff': 'yes',
                'sparse_ok': True,
                'tiled': True,
            })
            del sal_profile['nodata']
            sal_data = np.zeros((3, height, width), dtype=np.float32)

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
    if args.prediction:
        with torch.no_grad():
            for batch in tqdm.tqdm(batches):
                batch_stack = np.stack([
                    np.stack([ds.read(window=window) for ds in in_datasets])
                    for window in batch
                ]).astype(np.float32)
                batch_stack = torch.from_numpy(batch_stack).to(
                    dtype=torch.float32, device=device)
                raw = model(batch_stack)
                for i, window in enumerate(batch):
                    x = window.col_off
                    y = window.row_off
                    w = window.width
                    h = window.height
                    raw_data[:, y:(y + h), x:(x + w)] += raw[i, :, :, :].detach()

    if args.prediction:
        raw_data = raw_data.softmax(dim=0).cpu().numpy()
        out_data = np.expand_dims(np.argmax(raw_data, axis=0), axis=0)

    # Salience
    if args.salience:
        for param in model.parameters():
            param.requires_grad = False

        for batch in tqdm.tqdm(batches):
            batch_stack = np.stack([
                np.stack([ds.read(window=window) for ds in in_datasets])
                for window in batch
            ]).astype(np.float32)
            batch_stack = torch.from_numpy(batch_stack).to(
                dtype=torch.float32, device=device)
            batch_stack.requires_grad = True
            raw = model(batch_stack)
            score, _ = torch.max(raw, dim=-3)
            torch.sum(score).backward()
            sal = batch_stack * batch_stack.grad.abs()
            sal = torch.sum(sal[:, :, [4, 3, 2], :, :], dim=1)
            for i, window in enumerate(batch):
                x = window.col_off
                y = window.row_off
                w = window.width
                h = window.height
                sal_data[:, y:(y + h), x:(x + w)] += sal[i, :, :, :].detach().cpu().numpy()

    # Close r/o datasets
    for dataset in in_datasets:
        dataset.close()

    # Output results
    if args.prediction:
        log.info(f'Writing {raw_outfile} to disk')
        with rio.open(raw_outfile, 'w', **raw_profile) as ds:
            ds.write(raw_data)
        log.info(f'Writing {out_outfile} to disk')
        with rio.open(out_outfile, 'w', **out_profile) as ds:
            ds.write(out_data)
            ds.write_colormap(
                1, {
                    0: (0x00, 0xAE, 0xEF),
                    1: (0xEC, 0x00, 0x8C),
                    2: (0xFF, 0xEF, 0x00),
                    3: (0x00, 0x00, 0x00)
                })

    if args.salience:
        log.info(f'Writing {sal_outfile} to disk')
        with rio.open(sal_outfile, 'w', **sal_profile) as ds:
            ds.write(sal_data)
