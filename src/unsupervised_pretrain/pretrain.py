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
import sys

import numpy as np
import torch
import tqdm
from pytorch_metric_learning import losses
from torch.utils.data import DataLoader

from datasets import DigestDataset, SeriesDataset, SeriesEmbedDataset
from models import (Hat, SeriesEfficientNetb0, SeriesMobileNetv3,
                    SeriesResNet18, freeze, unfreeze)

if __name__ == "__main__":
    # yapf: disable
    # Command line arguments
    parser = argparse.ArgumentParser(description="Pretrain a model using a bunch unlabeled Sentinel-2 time series")
    parser.add_argument("cog_dirs", nargs="+", type=str, help="Paths to the data")
    parser.add_argument("--architecture", type=str, default="resnet18", choices=["resnet18", "mobilenetv3", "efficientnetb0"], help="The model architecture to use (default: resnet18)")
    parser.add_argument("--batch-size", type=int, default=6, help="The batch size (default: 6)")
    parser.add_argument("--dataset", type=str, default="series", choices=["embed-series", "series", "digest"], help="The type of data found in the data directories (default: series)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="The device to use for training (default: cuda)")
    parser.add_argument("--epochs", type=int, default=8, help="The number of epochs (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-3, help="The learning rate (default: 1e-3)")
    parser.add_argument('--no-pretrained', action='store_false', dest='pretrained', default=True, help='Whether to start from pretrained weights (default: True)')
    parser.add_argument("--num-workers", type=int, default=2, help="Number of worker processes for the DataLoader (default: 2)")
    parser.add_argument("--output-dir", type=str, default=".", help="The directory where logs and artifacts will be deposited (default: .)")
    parser.add_argument("--pth-in", type=str, help="Optional path to a .pth file to use as a starting point for model training")
    parser.add_argument("--pth-out", type=str, default="model.pth", help="The name of the output .pth file (default: model.pth)")
    parser.add_argument("--pth-hat1-in", type=str, help="Optional path to a .pth file to use as a starting point for training the first hat (\"embed-series\" only)")
    parser.add_argument("--pth-hat1-out", type=str, default="hat1.pth", help="The name of the output .pth file for the first hat (\"embed-series\" only, default: hat.pth)")
    parser.add_argument("--pth-hat2-in", type=str, help="Optional path to a .pth file to use as a starting point for training the second hat (\"embed-series\" only)")
    parser.add_argument("--pth-hat2-out", type=str, default="hat2.pth", help="The name of the output .pth file for the second hat (\"embed-series\" only, default: hat.pth)")
    parser.add_argument("--series-length", type=int, default=8, help="The number of time steps in each sample (default: 8)")
    parser.add_argument("--size", type=int, default=512, help="The tile size (default: 512)")
    # yapf: enable
    args = parser.parse_args()

    if args.pth_hat1_in is not None or args.pth_hat1_out != "hat1.pth" or args.pth_hat2_in is not None or args.pth_hat2_out != "hat2.pth":
        assert args.dataset == "embed-series"

    # Dataset and DataLoader for training set
    if args.dataset == "series":
        assert args.size % 64 == 0
        dataset = SeriesDataset(args.cog_dirs,
                                size=args.size,
                                series_length=args.series_length)
    elif args.dataset == "embed-series":
        assert args.size % 64 == 0
        dataset = SeriesEmbedDataset(args.cog_dirs,
                                     size=args.size,
                                     series_length=args.series_length)
    elif args.dataset == "digest":
        dataset = DigestDataset(args.cog_dirs)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    # Logging
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)-15s %(message)s")  # yapf: disable
    log = logging.getLogger()
    if args.output_dir is not None:
        fh = logging.FileHandler(f"{args.output_dir}/output.log")
        log.addHandler(fh)

    log.info(args.__dict__)

    # PyTorch device
    device = torch.device(args.device)

    # Model
    if args.pth_in is None:
        if args.architecture == "resnet18":
            model = SeriesResNet18(pretrained=args.pretrained).to(device)
        elif args.architecture == "mobilenetv3":
            model = SeriesMobileNetv3(pretrained=args.pretrained).to(device)
        elif args.architecture == "efficientnetb0":
            model = SeriesEfficientNetb0(pretrained=args.pretrained).to(device)
    elif args.pth_in:
        model = torch.load(args.pth_in, map_location=device).to(device)
        log.info(f"Model weights loaded from {args.pth_in}")

    if args.dataset == "embed-series":
        E = 768  # Instructor-XL
        if args.pth_hat1_in is not None:
            hat1 = torch.load(args.pth_hat1_in, map_location=device).to(device)
        if args.pth_hat2_in is not None:
            hat2 = torch.load(args.pth_hat2_in, map_location=device).to(device)
        else:
            hat1 = Hat(dim1=model.embedding_dim, dim2=64).to(device)
            hat2 = Hat(dim1=E, dim2=64).to(device)

    # Loss function, optimizer, scheduler
    base_obj = losses.TripletMarginLoss().to(device)
    obj1 = losses.SelfSupervisedLoss(base_obj).to(device)
    opt1 = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched1 = torch.optim.lr_scheduler.OneCycleLR(
        opt1,
        max_lr=args.lr,
        steps_per_epoch=len(dataloader),
        epochs=args.epochs)
    if args.dataset == "embed-series":
        obj2 = torch.nn.CosineEmbeddingLoss().to(device)
        params = list(hat1.parameters()) + list(hat2.parameters())
        opt2 = torch.optim.Adam(params, lr=args.lr)
        sched2 = torch.optim.lr_scheduler.OneCycleLR(
            opt2,
            max_lr=args.lr,
            steps_per_epoch=len(dataloader),
            epochs=args.epochs)

    target = torch.ones(args.batch_size * 2, device=device)

    for epoch in range(0, args.epochs):
        model.train()
        if args.dataset == "embed-series":
            hat1.train()
            hat2.train()

        training_losses1 = []
        training_losses2 = []
        for data in tqdm.tqdm(dataloader):

            if args.dataset == "embed-series":
                imagery_a, imagery_b, embeddings_text_a, embeddings_text_b = data
                embeddings_text_a = embeddings_text_a.to(device)
                embeddings_text_b = embeddings_text_b.to(device)
                embeddings_text = torch.cat([embeddings_text_a, embeddings_text_b], dim=0)
            else:
                imagery_a, imagery_b = data
            imagery_a = imagery_a.to(device)
            imagery_b = imagery_b.to(device)
            imagery = torch.cat([imagery_a, imagery_b], dim=0)

            opt1.zero_grad()
            if args.dataset == "embed-series":
                opt2.zero_grad()

            # Hats
            if args.dataset == "embed-series":
                if target.shape[0] != imagery.shape[0]:
                    target = torch.ones(imagery.shape[0], device=device)
                assert imagery.shape[0] == embeddings_text.shape[0]
                loss2 = obj2(hat1(model(imagery)), hat2(embeddings_text), target)
                training_losses2.append(loss2.item())
                loss2.backward()
                opt2.step()
                opt2.zero_grad()

            # Body
            loss1 = obj1(model(imagery_a), model(imagery_b.to(device)))
            training_losses1.append(loss1.item())
            loss1.backward()
            opt1.step()
            sched1.step()

            # Hats
            if args.dataset == "embed-series":
                if target.shape[0] != imagery.shape[0]:
                    target = torch.ones(imagery_a.shape[0], device=device)
                assert imagery.shape[0] == embeddings_text.shape[0]
                loss2 = obj2(hat1(model(imagery)), hat2(embeddings_text), target)
                training_losses2.append(loss2.item())
                loss2.backward()
                opt2.step()
                sched2.step()

        training_losses1 = np.mean(training_losses1)
        if args.dataset != "embed-series":
            log.info(f"epoch={epoch} training_loss={training_losses1}")
        else:
            training_losses2 = np.mean(training_losses2)
            log.info(f"epoch={epoch} training_loss1={training_losses1} training_loss2={training_losses2}")

        if args.output_dir is not None:
            model_save_path = f"{args.output_dir}/{args.pth_out}"
            model_save_path = model_save_path.replace(".pth", f"-{epoch}.pth")
            torch.save(model, model_save_path)
            if args.dataset == "embed-series":
                hat1_save_path = f"{args.output_dir}/{args.pth_hat1_out}"
                hat1_save_path = hat1_save_path.replace(".pth", f"-{epoch}.pth")
                torch.save(hat1, hat1_save_path)
                hat2_save_path = f"{args.output_dir}/{args.pth_hat2_out}"
                hat2_save_path = hat2_save_path.replace(".pth", f"-{epoch}.pth")
                torch.save(hat2, hat2_save_path)

    # Save the model after the last epoch if output_dir is provided
    if args.output_dir is not None:
        model_save_path = f"{args.output_dir}/{args.pth_out}"
        torch.save(model, model_save_path)
        log.info(f"Model saved to {model_save_path}")
        if args.dataset == "embed-series":
            hat1_save_path = f"{args.output_dir}/{args.pth_hat1_out}"
            torch.save(hat1, hat1_save_path)
            hat2_save_path = f"{args.output_dir}/{args.pth_hat2_out}"
            torch.save(hat2, hat2_save_path)
            log.info(f"Hats saved to {hat1_save_path} and {hat2_save_path}, respectively")
