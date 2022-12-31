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

from dataset import NpzSeriesDataset


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
    parser.add_argument('--architecture',
                        required=False,
                        type=str,
                        default='resnet18',
                        choices=[
                            'resnet18', 'resnet34', 'resnet50', 'resnet101',
                            'resnet152', 'cheaplab'
                        ])
    parser.add_argument('--pth', required=True, type=str)
    parser.add_argument('--num-workers', required=False, type=int, default=0)
    parser.add_argument('--input-dir', required=True, type=str)
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--size', required=False, type=int, default=256)
    return parser


if __name__ == '__main__':
    args = cli_parser().parse_args()
    logging.basicConfig(stream=sys.stderr,
                        level=logging.INFO,
                        format='%(asctime)-15s %(message)s')
    log = logging.getLogger()
    dataloader_cfg['batch_size'] = 1
    dataloader_cfg['num_workers'] = args.num_workers

    log.info(args)

    device = torch.device('cuda')
    if 'resnet' in args.architecture:
        model = torch.hub.load(
            'jamesmcclain/pytorch-fpn:02eb7d4a3b47db22ec30804a92713a08acff6af8',
            'make_fpn_resnet',
            name=args.architecture,
            fpn_type='panoptic',
            num_classes=6,
            fpn_channels=256,
            in_channels=12,
            out_size=(args.size, args.size)).to(device)
    else:
        raise Exception()
    model.load_state_dict(torch.load(args.pth), strict=False)
    model = model[0]
    model.eval()

    # Batches
    dl = torch.utils.data.DataLoader(NpzSeriesDataset(args.input_dir),
                                     **dataloader_cfg)
    for batch in tqdm.tqdm(dl):
        filename = f'{args.output_dir}/{str(uuid.uuid4())}.npz'
        source = model(batch[0][0].to(device))[-1].detach().cpu().numpy()
        target = model(batch[1].to(device))[-1].detach().cpu().numpy()
        np.savez_compressed(filename, source=source, target=target)
