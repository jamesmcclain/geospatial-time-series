# BSD 3-Clause License
#
# Copyright (c) 2022-23, Azavea, Element84, James McClain
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

from datasets import NpzSeriesDataset


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
