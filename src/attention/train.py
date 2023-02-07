#!/usr/bin/env python3

import argparse
import datetime
import logging
import math
import random
import sys

import numpy as np
import torch
import torchvision as tv
import tqdm
from PIL import Image

from datasets import (InMemorySeasonalDataset, NpzSeriesDataset,
                      RawSeriesDataset)
from models import (AttentionSegmenter,
                    AttentionSegmenterIn, AttentionSegmenterOut)

ARCHITECTURES = [
    'attention-segmenter',
    'attention-segmenter-in',
    'attention-segmenter-out',
]
DATASETS = ['in-memory-seasonal']
RESNETS = ['resnet18', 'resnet34', 'resnet50']


def worker_init_fn(i):
    seed = i + int(round(datetime.datetime.now().timestamp()))
    random.seed(seed)


dataloader_cfg = {
    'batch_size': None,
    'num_workers': None,
    'worker_init_fn': worker_init_fn,
    'shuffle': False,
}


def cli_parser():
    # yapf: disable
    parser = argparse.ArgumentParser()

    # Dataset, model type, input, output
    parser.add_argument('--architecture', required=True, type=str, choices=ARCHITECTURES)
    parser.add_argument('--dataset', required=True, type=str, choices=DATASETS)
    parser.add_argument('--output-dir', required=False, type=str)
    parser.add_argument('--resnet-architecture', required=False, type=str, choices=RESNETS)
    parser.add_argument('--resnet-state', required=False, type=str)
    parser.add_argument('--series', required=False, type=str, nargs='+')
    parser.add_argument('--size', required=False, type=int, default=256)
    parser.add_argument('--target', required=False, type=str)

    # Hyperparameters
    parser.add_argument('--eval-batches', required=False, type=int)
    parser.add_argument('--train-batches', required=False, type=int)
    parser.add_argument('--batch-size', required=False, type=int, default=4)

    parser.add_argument('--dimensions', required=False, type=int, default=512)
    parser.add_argument('--num-heads', required=False, type=int, default=1)

    parser.add_argument('--epochs', required=False, type=int, default=[13, 33], nargs='+')
    parser.add_argument('--gamma', required=False, type=float, default=0.7)

    parser.add_argument('--lr', required=False, type=float, default=1e-3)
    parser.add_argument('--clip', required=False, type=float, default=1)

    parser.add_argument('--sequence-limit', required=False, type=int, default=10)

    # Other
    parser.add_argument('--num-workers', required=False, type=int, default=1)
    parser.add_argument('--wandb-name', required=False, type=str, default=None)

    return parser
    # yapf: enable


if __name__ == '__main__':
    args = cli_parser().parse_args()
    logging.basicConfig(stream=sys.stderr,
                        level=logging.INFO,
                        format='%(asctime)-15s %(message)s')
    log = logging.getLogger()
    if args.output_dir is not None:
        fh = logging.FileHandler(f'{args.output_dir}/output.log')
        log.addHandler(fh)
    dataloader_cfg['batch_size'] = args.batch_size
    dataloader_cfg['num_workers'] = args.num_workers

    log.info(args.__dict__)

    try:
        if args.wandb_name is None:
            raise Exception('XXX')
        import wandb
        project = f'geospatial-time-series {args.wandb_name}'
        wandb.init(project=project,
                   config={
                       "learning_rate": args.lr,
                       "training_batches": args.train_batches,
                       "eval_batches": args.eval_batches,
                       "epochs": args.epochs,
                       "batch_size": args.batch_size,
                       "gamma": args.gamma,
                       "sequence_limit": args.sequence_limit,
                       "sequence_length": len(args.series),
                       "image_size": args.size,
                       "dimensions": args.dimensions,
                       "architecture": args.architecture,
                       "resnet_architecture": args.resnet_architecture,
                       "num_heads": args.num_heads,
                       "dataset": args.dataset,
                   })
    except:
        log.info('No wandb')

    # ------------------------------------------------------------------------

    if args.dataset == 'in-memory-seasonal':
        assert isinstance(args.series, list)
        assert isinstance(args.target, str)
        assert args.train_batches is not None
        assert args.eval_batches is not None

        bs = args.batch_size
        tb = args.train_batches
        eb = args.eval_batches
        nw = args.num_workers if args.num_workers > 0 else 1

        train_dl = torch.utils.data.DataLoader(
            InMemorySeasonalDataset(args.series,
                                    args.target,
                                    size=args.size,
                                    dimensions=args.dimensions,
                                    sequence_limit=args.sequence_limit,
                                    digest_labels='classifier' in args.architecture,
                                    evaluation=False),
            **dataloader_cfg,
        )
        eval_dl = torch.utils.data.DataLoader(
            InMemorySeasonalDataset(args.series,
                                    args.target,
                                    size=args.size,
                                    dimensions=args.dimensions,
                                    sequence_limit=args.sequence_limit,
                                    digest_labels='classifier' in args.architecture,
                                    evaluation=True),
            **dataloader_cfg,
        )

        train_dl = iter(train_dl)
        eval_dl = iter(eval_dl)

    # ------------------------------------------------------------------------

    device = torch.device('cuda')

    if args.architecture == 'attention-segmenter':
        assert args.resnet_architecture is not None
        assert args.resnet_state is not None
        assert args.dimensions is not None
        model = AttentionSegmenter(
            args.resnet_architecture,
            args.resnet_state,
            args.size,
            args.dimensions,
            num_heads=args.num_heads,
        ).to(device)
    elif args.architecture == 'attention-segmenter-in':
        assert args.resnet_architecture is not None
        assert args.resnet_state is not None
        assert args.dimensions is not None
        model = AttentionSegmenterIn(
            args.resnet_architecture,
            args.resnet_state,
            args.size,
            args.dimensions,
        ).to(device)
    elif args.architecture == 'attention-segmenter-out':
        assert args.resnet_architecture is not None
        assert args.resnet_state is not None
        assert args.dimensions is not None
        model = AttentionSegmenterOut(
            args.resnet_architecture,
            args.resnet_state,
            args.size,
            args.dimensions,
        ).to(device)

    obj = torch.nn.CrossEntropyLoss(
        weight=torch.Tensor([1., 1., 1., 1.]),
        ignore_index=-1,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=args.gamma)

    # ------------------------------------------------------------------------

    best = math.inf
    for phase in range(2):

        if phase == 0:
            model.freeze_resnet()
            log.info('ResNet frozen')
        if phase != 0:
            model.unfreeze_resnet()
            log.info('ResNet unfrozen')

        for epoch in range(1, args.epochs[phase] + 1):
            loss_t = []
            loss_e = []

            # Train
            model.train()
            for _ in tqdm.tqdm(range(0, args.train_batches), desc=f'Epoch {epoch}: training'):
                opt.zero_grad()

                batch = next(train_dl)
                x = batch[0].to(device)
                pos = batch[2].to(device)
                target = batch[1].to(device)
                out = model(x, pos)
                loss = obj(out, target)
                loss_t.append(loss.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.clip,
                )
                opt.step()
                sched.step()
            loss_t = np.mean(loss_t)

            # Evaluation
            model.eval()
            batches = args.eval_batches
            with torch.no_grad():
                for _ in tqdm.tqdm(range(0, args.eval_batches), desc=f'Epoch {epoch}: evaluation'):
                    batch = next(eval_dl)
                    x = batch[0].to(device)
                    pos = batch[2].to(device)
                    target = batch[1].to(device)
                    out = model(x, pos)
                    loss = obj(out, target)
                    loss_e.append(loss.item())
            loss_e = np.mean(loss_e)

            # yapf: disable
            if loss_e < best:
                best = loss_e
                log.info(f'✓ Epoch={epoch} train={loss_t} eval={loss_e}')
                if args.output_dir:
                    torch.save(model.state_dict(), f'{args.output_dir}/{args.architecture}-{args.resnet_architecture}-best.pth')
            else:
                log.info(f'✗ Epoch={epoch} train={loss_t} eval={loss_e}')
            # yapf: enable

            try:
                wandb_dict = {
                    "loss train": loss_t,
                    "loss eval": loss_e,
                }
            except:
                pass

    # yapf: disable
    if args.output_dir:
        torch.save(model.state_dict(), f'{args.output_dir}/{args.architecture}-{args.resnet_architecture}-last.pth')
    # yapf: enable
