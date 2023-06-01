#!/usr/bin/env python3

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

from datasets import RawSeriesDataset


def worker_init_fn(i):
    seed = i + int(round(datetime.datetime.now().timestamp()))
    random.seed(seed)


dataloader_cfg = {
    'batch_size': None,
    'num_workers': None,
    'worker_init_fn': worker_init_fn,
}


def cli_parser():
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', required=False, type=int, default=1)
    parser.add_argument('--train-batches', required=False, type=int, default=8)
    parser.add_argument('--eval-batches', required=False, type=int, default=2)
    parser.add_argument('--max-sequence', required=False, type=int, default=-1)
    parser.add_argument('--mosaic', required=True, type=str)
    parser.add_argument('--num-workers', required=False, type=int, default=0)
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--series', required=True, type=str, nargs='+')
    parser.add_argument('--size', required=False, type=int, default=256)
    parser.add_argument('--channels', required=False, type=int, nargs='+', default=None)
    parser.add_argument('--noop', required=False, type=int)
    return parser
    # yapf: enable


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
            RawSeriesDataset(args.series,
                             args.mosaic,
                             args.size,
                             args.max_sequence,
                             evaluation=False,
                             channels=args.channels),
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
            RawSeriesDataset(args.series,
                             args.mosaic,
                             args.size,
                             args.max_sequence,
                             evaluation=True,
                             channels=args.channels),
            **dataloader_cfg,
        ))
    for _ in range(0, args.eval_batches):
        batch = next(eval_dl)
        filename = f'{args.output_dir}/eval/{str(uuid.uuid4())}.npz'
        np.savez_compressed(filename, source=batch[0], target=batch[1])
    del eval_dl
