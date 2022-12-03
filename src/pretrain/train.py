#!/usr/bin/env python3

import argparse
import logging
import torch
import torchvision as tv
import tqdm

from dataset import TLDataset

dataloader_cfg = {
    'batch_size': None,
    'num_workers': None,
}


def cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', required=False, type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--batch-size', required=False, type=int, default=8)
    parser.add_argument('--epochs', required=False, type=int, default=2**7)
    parser.add_argument('--num-workers', required=False, type=int, default=8)
    parser.add_argument('--path', required=True, type=str)
    parser.add_argument('--size', required=False, type=int, default=256)
    parser.add_argument('--train-batches', required=False, type=int, default=2**12)
    parser.add_argument('--eval-batches', required=False, type=int, default=2**9)
    return parser


if __name__ == '__main__':
    args = cli_parser().parse_args()
    dataloader_cfg['batch_size'] = args.batch_size
    dataloader_cfg['num_workers'] = args.num_workers

    train_dl = iter(torch.utils.data.DataLoader(
        TLDataset(args.path, size=args.size, evaluation=False),
        **dataloader_cfg,
    ))
    eval_dl = iter(torch.utils.data.DataLoader(
        TLDataset(args.path, size=args.size, evaluation=True),
        **dataloader_cfg,
    ))

    for i in tqdm.tqdm(range(0, args.epochs)):
        for j in range(0, args.train_batches):
            batch = next(train_dl)
        for j in range(0, args.eval_batches):
            batch = next(eval_dl)
