#!/usr/bin/env python3

import argparse
import datetime
import logging
import math
import random
import sys
import uuid

import numpy as np
import torch
import torchvision as tv
import tqdm

from dataset import SeriesDataset


def worker_init_fn(i):
    seed = i + int(round(datetime.datetime.now().timestamp()))
    random.seed(seed)


dataloader_cfg = {
    'batch_size': None,
    'num_workers': None,
    'worker_init_fn': worker_init_fn,
}


def cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', required=False, type=int, default=1)
    parser.add_argument('--train-batches',
                        required=False,
                        type=int,
                        default=8)
    parser.add_argument('--eval-batches',
                        required=False,
                        type=int,
                        default=2)
    parser.add_argument('--max-sequence', required=False, type=int, default=-1)
    parser.add_argument('--mosaic', required=True, type=str)
    parser.add_argument('--num-workers', required=False, type=int, default=0)
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--series', required=True, type=str, nargs='+')
    parser.add_argument('--size', required=False, type=int, default=256)
    return parser


if __name__ == '__main__':
    args = cli_parser().parse_args()
    logging.basicConfig(stream=sys.stderr,
                        level=logging.INFO,
                        format='%(asctime)-15s %(message)s')
    log = logging.getLogger()
    dataloader_cfg['batch_size'] = args.batch_size
    dataloader_cfg['num_workers'] = args.num_workers

    # Training batches
    train_dl = iter(
        torch.utils.data.DataLoader(
            SeriesDataset(args.series,
                          args.mosaic,
                          args.size,
                          args.max_sequence,
                          evaluation=False),
            **dataloader_cfg,
        ))
    for _ in range(0, args.train_batches):
        batch = next(train_dl)
        filename = f'{args.output_dir}/train/{str(uuid.uuid4())}.npz'
        np.savez_compressed(filename, source=batch[0], target=batch[1])
    del train_dl

    # Eval batches
    eval_dl = iter(
        torch.utils.data.DataLoader(
            SeriesDataset(args.series,
                          args.mosaic,
                          args.size,
                          args.max_sequence,
                          evaluation=True),
            **dataloader_cfg,
        ))
    for _ in range(0, args.eval_batches):
        batch = next(eval_dl)
        filename = f'{args.output_dir}/eval/{str(uuid.uuid4())}.npz'
        np.savez_compressed(filename, source=batch[0], target=batch[1])
    del eval_dl
