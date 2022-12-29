#!/usr/bin/env python3

import argparse
import logging
import math
import random
import sys

import torch
import torchvision as tv
import tqdm

from dataset import SeriesDataset


def worker_init_fn(i):
    random.seed(i)


dataloader_cfg = {
    'batch_size': None,
    'num_workers': None,
    'worker_init_fn': worker_init_fn,
}


def cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture',
                        required=False,
                        type=str,
                        default='resnet18',
                        choices=[
                            'resnet18', 'resnet34', 'resnet50', 'resnet101',
                            'resnet152', 'cheaplab'
                        ])
    parser.add_argument('--batch-size', required=False, type=int, default=8)
    parser.add_argument('--epochs', required=False, type=int, default=2**7)
    parser.add_argument('--eval-batches',
                        required=False,
                        type=int,
                        default=2**6)
    parser.add_argument('--max-sequence', required=False, type=int, default=20)
    parser.add_argument('--mosaic', required=True, type=str)
    parser.add_argument('--num-workers', required=False, type=int, default=8)
    parser.add_argument('--output-dir', required=False, type=str, default=None)
    parser.add_argument('--series', required=True, type=str, nargs='+')
    parser.add_argument('--size', required=False, type=int, default=256)
    parser.add_argument('--train-batches',
                        required=False,
                        type=int,
                        default=2**9)
    return parser


if __name__ == '__main__':
    args = cli_parser().parse_args()
    logging.basicConfig(stream=sys.stderr,
                        level=logging.INFO,
                        format='%(asctime)-15s %(message)s')
    log = logging.getLogger()
    if args.output_dir is not None:
        fh = logging.FileHandler(f'{args.output_dir}/{args.architecture}.log')
        log.addHandler(fh)
    dataloader_cfg['batch_size'] = args.batch_size
    dataloader_cfg['num_workers'] = args.num_workers

    train_dl = iter(
        torch.utils.data.DataLoader(
            SeriesDataset(args.series,
                          args.mosaic,
                          args.size,
                          args.max_sequence,
                          evaluation=False),
            **dataloader_cfg,
        ))
    eval_dl = iter(
        torch.utils.data.DataLoader(
            SeriesDataset(args.series,
                          args.mosaic,
                          args.size,
                          args.max_sequence,
                          evaluation=True),
            **dataloader_cfg,
        ))

    batch = next(train_dl)

    import pdb ; pdb.set_trace()

    pass
