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
from models import (AttentionClassifier, BaselineClassifier, EntropyLoss,
                    ResnetTransformerClassifier)


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
    parser.add_argument('--architecture', required=True, type=str, choices=['baseline-classifier', 'attention-classifier', 'resnet-transformer-classifier'])
    parser.add_argument('--dataset', required=True, type=str, choices=['in-memory-seasonal'])
    parser.add_argument('--input-dir', required=False, type=str)
    parser.add_argument('--output-dir', required=False, type=str)
    parser.add_argument('--resnet-architecture', required=False, type=str, choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--resnet-state', required=False, type=str)
    parser.add_argument('--series', required=False, type=str, nargs='+')
    parser.add_argument('--size', required=False, type=int, default=256)
    parser.add_argument('--target', required=False, type=str)

    # Hyperparameters
    parser.add_argument('--eval-batches', required=False, type=int)
    parser.add_argument('--train-batches', required=False, type=int)
    parser.add_argument('--batch-size', required=False, type=int, default=16)

    parser.add_argument('--dimensions', required=False, type=int, default=512)
    parser.add_argument('--encoder-layers', required=False, type=int, default=1)
    parser.add_argument('--num-heads', required=False, type=int, default=1)

    # parser.add_argument('--entropy', dest='entropy', default=False, action='store_true')
    parser.add_argument('--epochs', required=False, type=int, default=2**7)
    parser.add_argument('--gamma', required=False, type=float, default=0.7)
    parser.add_argument('--lr', required=False, type=float, default=3e-4)

    parser.add_argument('--sequence-limit', required=False, type=int, default=10)
    parser.add_argument('--train-ipi', required=False, type=int, default=None)
    parser.add_argument('--eval-ipi', required=False, type=int, default=None)

    # Other
    parser.add_argument('--num-workers', required=False, type=int, default=1)

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
        import wandb
        wandb.init(project="geospatial-time-series",
                   config={
                       "learning_rate": args.lr,
                       "training_batches": args.train_batches,
                       "eval_batches": args.eval_batches,
                       "epochs": args.epochs,
                       "batch_size": args.batch_size,
                       "gamma": args.gamma,
                       "sequence_limit": args.sequence_limit,
                       "image_size": args.size,
                       "dimensions": args.dimensions,
                       "architecture": args.architecture,
                       "resnet_architecture": args.resnet_architecture,
                       "transformer_encoder_layers": args.encoder_layers,
                       "transformer_num_heads": args.num_heads,
                       "dataset": args.dataset,
                       "train_ipi": args.train_ipi,
                       "eval_ipi": args.eval_ipi,
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
        if args.train_ipi is None:
            train_ipi = (bs * tb) // nw
        else:
            train_ipi = args.train_ipi
        if args.eval_ipi is None:
            eval_ipi = (bs * eb) // nw
        else:
            eval_ipi = args.eval_ipi

        train_dl = torch.utils.data.DataLoader(
            InMemorySeasonalDataset(args.series,
                                    args.target,
                                    iters_per_incr=train_ipi,
                                    size=args.size,
                                    dimensions=512,
                                    sequence_limit=args.sequence_limit,
                                    digest_labels=True,
                                    evaluation=False),
            **dataloader_cfg,
        )
        eval_dl = torch.utils.data.DataLoader(
            InMemorySeasonalDataset(args.series,
                                    args.target,
                                    iters_per_incr=eval_ipi,
                                    size=args.size,
                                    dimensions=512,
                                    sequence_limit=args.sequence_limit,
                                    digest_labels=True,
                                    evaluation=True),
            **dataloader_cfg,
        )

    if args.dataset == 'in-memory-seasonal':
        train_dl = iter(train_dl)
        eval_dl = iter(eval_dl)

    # ------------------------------------------------------------------------

    _, clss = next(train_dl)[1].shape
    device = torch.device('cuda')
    if args.architecture == 'series-resnet-classifier':
        assert args.resnet_architecture is not None
        assert args.resnet_state is not None
        assert args.dimensions is not None
        assert args.num_heads is not None
        assert args.encoder_layers is not None
        model = ResnetTransformerClassifier(
            args.resnet_architecture,
            args.resnet_state,
            args.size,
            # args.dimensions + 512,
            args.dimensions + 0,
            args.num_heads,
            args.encoder_layers,
            clss = clss,
        ).to(device)
    elif args.architecture == 'attention-classifier':
        assert args.resnet_architecture is not None
        assert args.resnet_state is not None
        assert args.dimensions is not None
        model = AttentionClassifier(
            args.resnet_architecture,
            args.resnet_state,
            args.size,
            args.dimensions,
            clss = clss,
        ).to(device)
    elif args.architecture == 'baseline-classifier':
        assert args.resnet_architecture is not None
        assert args.resnet_state is not None
        assert args.dimensions is not None
        model = BaselineClassifier(
            args.resnet_architecture,
            args.resnet_state,
            args.size,
            args.dimensions,
            clss = clss,
        ).to(device)

    obj1 = torch.nn.MSELoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=args.gamma)

    # ------------------------------------------------------------------------

    best = math.inf
    for epoch in range(1, args.epochs + 1):

        current = 0
        for mode in ['train', 'eval']:
            loss_float = []
            if mode == 'train':
                model.train()
                batches = args.train_batches
                for _ in tqdm.tqdm(range(0, batches),
                                   desc=f'Epoch {epoch}: training'):
                    batch = next(train_dl)
                    x = batch[0].to(device)
                    pos = batch[2].to(device)
                    target = batch[1].to(device)
                    # yapf: disable
                    if args.architecture in {'attention-classifier', 'resnet-transformer-classifier'}:
                        out = model(x, pos)
                    elif args.architecture in {'baseline-classifier'}:
                        out = model(x)
                    # yapf: enable
                    loss = obj1(out, target)
                    loss_float.append(loss.item())

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            elif mode == 'eval':
                model.eval()
                batches = args.eval_batches
                gts = []
                preds = []
                with torch.no_grad():
                    for _ in tqdm.tqdm(range(0, batches),
                                       desc=f'Epoch {epoch}: evaluation'):
                        batch = next(eval_dl)
                        x = batch[0].to(device)
                        pos = batch[2].to(device)
                        target = batch[1].to(device)
                        # yapf: disable
                        if args.architecture in {'attention-classifier', 'resnet-transformer-classifier'}:
                            out = model(x, pos)
                        elif args.architecture in {'baseline-classifier'}:
                            out = model(x)
                        # yapf: enable
                        gt = batch[1].detach().cpu().numpy()
                        pred = out.detach().cpu().numpy()
                        gts.append(gt)
                        preds.append(pred)
                        loss = obj1(out, target)
                        loss_float.append(loss.item())

                gts = np.concatenate(gts, axis=0)
                preds = np.concatenate(preds, axis=0)
                diffs = (gts - preds)
                mus = np.mean(diffs, axis=0)
                absmus = np.mean(np.absolute(diffs), axis=0)
                sigmas = np.sqrt(np.mean(np.power(diffs, 2), axis=0))
                gts = np.mean(gts, axis=0)
                preds = np.mean(preds, axis=0)

            loss_float = np.mean(loss_float)

            if mode == 'train':
                loss_t = loss_float
            elif mode == 'eval':
                loss_e = loss_float

        # yapf: disable
        if loss_e < best:
            best = loss_e
            log.info(
                f'✓ Epoch={epoch} train={loss_t} eval={loss_e} '
                f'gt={gts} pred={preds} μ={mus} σ={sigmas}'
            )
            if args.output_dir:
                torch.save(model.state_dict(), f'{args.output_dir}/{args.architecture}-{args.resnet_architecture}-best.pth')
        else:
            log.info(
                f'✗ Epoch={epoch} train={loss_t} eval={loss_e} '
                f'gt={gts} pred={preds} μ={mus} σ={sigmas}'
            )
        # yapf: enable
        try:
            wandb_dict = {
                "loss train": loss_t,
                "loss eval": loss_e,
            }
            for i in range(len(mus)):
                wandb_dict.update({f'μ{i}': mus[i]})
                wandb_dict.update({f'abs μ{i}': absmus[i]})
                wandb_dict.update({f'σ{i}': sigmas[i]})
            wandb.log(wandb_dict)
        except:
            pass

    # yapf: disable
    if args.output_dir:
        torch.save(model.state_dict(), f'{args.output_dir}/{args.architecture}-{args.resnet_architecture}-last.pth')
    # yapf: enable
