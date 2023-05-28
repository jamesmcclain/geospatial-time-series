#!/usr/bin/env python3

import argparse
import logging
import math
import sys

import numpy as np
import torch
import torchvision as tv
from models import (SeriesEfficientNetb0, SeriesMobileNetv3, SeriesResNet18,
                    freeze, unfreeze)
from pytorch_metric_learning import losses, miners
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import SeriesDataset

if __name__ == "__main__":
    # yapf: disable
    # Command line arguments
    parser = argparse.ArgumentParser(description='Pretrain a model using a bunch unlabeled Sentinel-2 time series')
    parser.add_argument('cog_dirs', nargs='+', type=str, help='Path to the training set tarball')
    parser.add_argument('--architecture', type=str, default='resnet18', choices=['resnet18', 'mobilenetv3', 'efficientnetb0'], help='The model architecture to use (default: resnet18)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (default: 8)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='The device to use for training (default: cuda)')
    parser.add_argument('--epochs', type=int, default=8, help='The number of epochs (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-3, help='The learning rate (default: 1e-3)')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of worker processes for the DataLoader (default: 2)')
    parser.add_argument('--output-dir', type=str, default='.', help='The directory where logs and artifacts will be deposited (default: .)')
    parser.add_argument('--pth-out', type=str, default='model.pth', help='The name of the output .pth file (default: model.pth)')
    parser.add_argument('--pth-in', type=str, help='Optional path to a .pth file to use as a starting point for model training')
    # yapf: enable
    args = parser.parse_args()

    # Dataset and DataLoader for training set
    dataset = SeriesDataset(args.cogs)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    # Logging
    logging.basicConfig(stream=sys.stderr,
                        level=logging.INFO,
                        format='%(asctime)-15s %(message)s')
    log = logging.getLogger()
    if args.output_dir is not None:
        fh = logging.FileHandler(f'{args.output_dir}/output.log')
        log.addHandler(fh)

    log.info(args.__dict__)

    # PyTorch device
    device = torch.device(args.device)

    # Model
    if args.pth_in is None:
        if args.architecture == 'resnet18':
            model = SeriesResNet18().to(device)
        elif args.architecture == 'mobilenetv3':
            model = SeriesMobileNetv3().to(device)
        elif args.architecture == 'efficientnetb0':
            model = SeriesEfficientNetb0().to(device)
    elif args.start_from:
        model = torch.load(args.pth_in, map_location=device).to(device)
        log.info(f"Model weights loaded from {args.pth_in}")

    # Loss function, optimizer, miner
    base_obj = losses.TripletMarginLoss().to(device)
    obj = losses.SelfSupervisedLoss(base_obj).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    miner = miners.BatchEasyHardMiner(pos_strategy="semihard",
                                      neg_strategy="easy")

    model.train()
    for epoch in range(0, args.epochs):
        for i, data in enumerate(dataloader, 0):
            imagery_a, imagery_b = data
            opt.zero_grad()
            embeddings_a = model(imagery_a.to(device))
            embeddings_b = model(imagery_b.to(device))
            good_pairs = miner(embeddings_a, embeddings_b).to(device)
            loss = obj(embeddings_a, embeddings_b, good_pairs)
            loss.backward()
            losses.append(loss.item())
            opt.step()
        losses = np.mean(losses)
        log.info(f"epoch={epoch} train_loss={losses}")

    # Save the model after the last epoch if output_dir is provided
    if args.output_dir is not None:
        model_save_path = f"{args.output_dir}/{args.pth_out}"
        torch.save(model, model_save_path)
        log.info(f"Model saved to {model_save_path}")
