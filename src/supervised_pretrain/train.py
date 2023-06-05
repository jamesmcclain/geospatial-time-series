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
import logging
import math
import random
import sys

import torch
import torchvision as tv
import tqdm

from datasets import TLDataset


def worker_init_fn(i):
    random.seed(i)


dataloader_cfg = {
    'batch_size': None,
    'num_workers': None,
    'worker_init_fn': worker_init_fn,
}


def cli_parser():
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', required=False, type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'cheaplab'])
    parser.add_argument('--batch-size', required=False, type=int, default=8)
    parser.add_argument('--epochs', required=False, type=int, default=42)
    parser.add_argument('--eval-batches', required=False, type=int, default=2**6)
    parser.add_argument('--num-workers', required=False, type=int, default=8)
    parser.add_argument('--output-dir', required=False, type=str, default=None)
    parser.add_argument('--input-dir', required=True, type=str)
    parser.add_argument('--size', required=False, type=int, default=256)
    parser.add_argument('--train-batches', required=False, type=int, default=2**9)
    return parser
    # yapf: enable


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
            TLDataset(args.input_dir, size=args.size, evaluation=False),
            **dataloader_cfg,
        ))
    eval_dl = iter(
        torch.utils.data.DataLoader(
            TLDataset(args.input_dir, size=args.size, evaluation=True),
            **dataloader_cfg,
        ))

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
    elif 'cheaplab' in args.architecture:
        model = torch.hub.load(
            'jamesmcclain/CheapLab:38af8e6cd084fc61792f29189158919c69d58c6a',
            'make_cheaplab_model',
            num_channels=12,
            out_channels=6).to(device)
    obj = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

    best = math.inf
    for i in range(0, args.epochs):

        model.train()
        loss_t = 0.0
        for j in tqdm.tqdm(range(0, args.train_batches)):
            batch = next(train_dl)
            out = model(batch[0].to(device))
            loss = obj(out, batch[1].to(device))
            loss_t += loss.item()
            loss.backward()
            opt.step()
            opt.zero_grad()
        loss_t /= float(args.train_batches)

        model.eval()
        loss_e = 0.0
        with torch.no_grad():
            for j in tqdm.tqdm(range(0, args.eval_batches)):
                batch = next(eval_dl)
                out = model(batch[0].to(device))
                loss_e = loss_e + obj(out, batch[1].to(device)).item()
            loss_e /= float(args.eval_batches)

        log.info(f'Epoch={i} train={loss_t} eval={loss_e}')
        if loss_e < best:
            best = loss_e
            if args.output_dir:
                torch.save(model.state_dict(),
                           f'{args.output_dir}/{args.architecture}-best.pth')

    if args.output_dir:
        torch.save(model.state_dict(),
                   f'{args.output_dir}/{args.architecture}-last.pth')
