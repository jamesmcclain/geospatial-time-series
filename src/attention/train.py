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

from datasets import InMemorySeasonalDataset
from models import AttentionSegmenter, CheaplabSegmenter, EntropyLoss

ARCHITECTURES = ['attention-segmenter', 'cheaplab-segmenter']
DATASETS = ['in-memory-seasonal']
RESNETS = ['resnet18', 'resnet34']


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

    parser.add_argument('--device', required=False, type=str, default='cuda', choices=['cuda', 'cpu'])

    # Dataset, model type, input, output
    parser.add_argument('--architecture', required=True, type=str, choices=ARCHITECTURES)
    parser.add_argument('--dataset', required=True, type=str, choices=DATASETS)
    parser.add_argument('--model-state', required=False, type=str, default=None)
    parser.add_argument('--output-dir', required=False, type=str)
    parser.add_argument('--resnet-architecture', required=False, type=str, choices=RESNETS)
    parser.add_argument('--resnet-state', required=False, type=str, default=None)
    parser.add_argument('--series', required=True, type=str, nargs='+')
    parser.add_argument('--size', required=False, type=int, default=256)
    parser.add_argument('--target', required=False, type=str)

    # Hyperparameters
    parser.add_argument('--batch-size', required=False, type=int, default=4)
    parser.add_argument('--eval-batches', required=False, type=int)
    parser.add_argument('--train-batches', required=False, type=int)

    parser.add_argument('--dimensions', required=False, type=int, default=512)
    parser.add_argument('--dropout', required=False, type=float, default=0.10)
    parser.add_argument('--num-heads', required=False, type=int, default=3)

    parser.add_argument('--clip', required=False, type=float, default=None)
    parser.add_argument('--epochs', required=False, type=int, default=[13, 7], nargs='+')
    parser.add_argument('--gamma', required=False, type=float, default=[0.837678, 0.719686], nargs='+')
    parser.add_argument('--lr', required=False, type=float, default=[1e-5, 1e-5],nargs='+')
    parser.add_argument('--phases', required=False, type=int, default=2)

    parser.add_argument('--sequence-limit', required=False, type=int, default=72)
    parser.add_argument('--howmuch', required=False, type=float, default=1.0)

    # Other
    parser.add_argument('--num-workers', required=False, type=int, default=5)
    parser.add_argument('--wandb-name', required=False, type=str, default=None)

    return parser
    # yapf: enable


class SpecialLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.cross = torch.nn.CrossEntropyLoss(ignore_index=-1)
        # self.bce = torch.nn.BCEWithLogitsLoss()
        # # self.entropy = EntropyLoss(mu=0., sigma=1.)
        # self.entropy = EntropyLoss()

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        y = torch.nn.functional.softmax(x, dim=1)

        loss = 0.

        # F1 scores
        for i in [0, 1, 2]:
            total_pos = torch.sum(y[:, i, :, :] * (target == i).float())
            gt_pos = torch.sum((target == i).float())
            pred_pos = torch.sum(y[:, i, :, :])
            recall = total_pos / (gt_pos + 1e-6)
            precision = total_pos / (pred_pos + 1e-6)
            loss -= recall
            loss -= precision
            # loss -= 2. / ((1. / recall) + (1. / precision))
        loss /= 6.

        # Segmentation correctness
        loss += self.cross(x, target)

        return loss


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

    class_names = ["other", "farm", "forest", "road"]
    log.info(args.__dict__)

    assert args.phases >= len(args.epochs)
    # assert args.phases >= len(args.gamma)

    device = torch.device(args.device)

    try:
        if args.wandb_name is None:
            raise Exception()
        import wandb
        project = f'geospatial-time-series {args.wandb_name}'
        wandb.init(project=project,
                   config={
                       "device": args.device,
                       "architecture": args.architecture,
                       "dataset": args.dataset,
                       "image_size": args.size,
                       "model_state": args.model_state,
                       "resnet_architecture": args.resnet_architecture,
                       "resnet_state": args.resnet_state,
                       "series_length": len(args.series),
                       "target": args.target.split('/')[-1],
                       "batch_size": args.batch_size,
                       "eval_batches": args.eval_batches,
                       "train_batches": args.train_batches,
                       "dimensions": args.dimensions,
                       "dropout": args.dropout,
                       "num_heads": args.num_heads,
                       "clip": args.clip,
                       "epochs": args.epochs,
                       "gamma": args.gamma,
                       "lr": args.lr,
                       "phases": args.phases,
                       "sequence_limit": args.sequence_limit,
                       "howmuch": args.howmuch,
                       "num_workers": args.num_workers,
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
                                    howmuch=args.howmuch,
                                    evaluation=False),
            **dataloader_cfg,
        )
        eval_dl = torch.utils.data.DataLoader(
            InMemorySeasonalDataset(args.series,
                                    args.target,
                                    size=args.size,
                                    dimensions=args.dimensions,
                                    sequence_limit=args.sequence_limit,
                                    howmuch=args.howmuch,
                                    evaluation=True),
            **dataloader_cfg,
        )

        train_dl = iter(train_dl)
        eval_dl = iter(eval_dl)

    # ------------------------------------------------------------------------

    if args.architecture == 'attention-segmenter':
        assert args.resnet_architecture is not None
        model = AttentionSegmenter(
            args.resnet_architecture,
            args.resnet_state,
            args.size,
            num_heads=args.num_heads,
            dropout=args.dropout,
        )
    elif args.architecture == 'cheaplab-segmenter':
        model = CheaplabSegmenter(num_heads=args.num_heads, )
    else:
        pass

    if args.model_state is not None:
        log.info(f'Loading state from {args.model_state}')
        model.load_state_dict(torch.load(args.model_state,
                                         map_location=torch.device('cpu')),
                              strict=True)

    model = model.to(device)

    obj = SpecialLoss().to(device)

    # ------------------------------------------------------------------------

    best = -math.inf
    for phase in range(args.phases):

        gamma = args.gamma[phase % len(args.gamma)]
        lr = args.lr[phase % len(args.lr)]
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=gamma)

        if phase % 2 == 1:
            model.freeze_resnet()
            log.info(f'ResNet frozen lr={lr}')
        elif phase % 2 == 0:
            model.unfreeze_resnet()
            log.info(f'ResNet unfrozen lr={lr}')

        for epoch in range(1, args.epochs[phase % len(args.epochs)] + 1):
            loss_t = []
            loss_e = []

            # Train
            model.train()
            for _ in tqdm.tqdm(range(0, args.train_batches),
                               desc=f'Epoch {epoch}: training'):
                opt.zero_grad()

                batch = next(train_dl)
                x = batch[0].to(device)
                pos = batch[2].to(device)
                target = batch[1].to(device)
                out = model(x, pos)
                loss = obj(out, target)
                loss_t.append(loss.item())

                loss.backward()
                if args.clip is not None:
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
            predictions = []
            groundtruth = []
            recall = []
            precision = []
            f1 = []
            with torch.no_grad():
                for _ in tqdm.tqdm(range(0, args.eval_batches),
                                   desc=f'Epoch {epoch}: evaluation'):
                    batch = next(eval_dl)
                    x = batch[0].to(device)
                    pos = batch[2].to(device)
                    target = batch[1].to(device)
                    out = model(x, pos)
                    loss = obj(out, target)
                    loss_e.append(loss.item())
                    predictions.append(
                        np.argmax(out.detach().cpu().numpy(),
                                  axis=1).flatten())
                    groundtruth.append(
                        batch[1].detach().cpu().numpy().flatten())
            loss_e = np.mean(loss_e)
            predictions = np.concatenate(predictions)
            groundtruth = np.concatenate(groundtruth)
            accuracy = (predictions == groundtruth).mean()
            for i in range(4):
                # yapf: disable
                _recall = ((predictions == i) * (groundtruth == i)).sum() / ((groundtruth == i).sum() + 1e-6) + 1e-6
                _precision = ((predictions == i) * (groundtruth == i)).sum() / ((predictions == i).sum() + 1e-6) + 1e-6
                _f1 = 2.0 / ((1 / _recall) + (1 / _precision))
                recall.append(_recall)
                precision.append(_precision)
                f1.append(_f1)
                # yapf: enable

            # yapf: disable
            if f1[1] > best:
                best = f1[1]
                log.info(f'✓ Epoch={epoch} train={loss_t} eval={loss_e} accuracy={accuracy} farm_recall={recall[1]} farm_precision={precision[1]} farm_f1={f1[1]}')
                if args.output_dir:
                    torch.save(model.state_dict(), f'{args.output_dir}/{args.architecture}-{args.resnet_architecture}-best.pth')
            else:
                log.info(f'✗ Epoch={epoch} train={loss_t} eval={loss_e} accuracy={accuracy} farm_recall={recall[1]} farm_precision={precision[1]} farm_f1={f1[1]}')
            # yapf: enable

            try:
                wandb_dict = {
                    "loss train": loss_t,
                    "loss eval": loss_e,
                    "overall accuracy": accuracy,
                }
                for i, name in enumerate(class_names):
                    wandb_dict.update({
                        f'{name} recall': recall[i],
                        f'{name} precision': precision[i],
                        f'{name} f1': f1[i],
                    })
                wandb.log(wandb_dict)
            except:
                pass

    try:
        conf_mat = wandb.plot.confusion_matrix(probs=None,
                                               y_true=groundtruth,
                                               preds=predictions,
                                               class_names=class_names)
        wandb.log({"conf_mat": conf_mat})
    except:
        pass

    # yapf: disable
    if args.output_dir:
        torch.save(model.state_dict(), f'{args.output_dir}/{args.architecture}-{args.resnet_architecture}-last.pth')
    # yapf: enable
