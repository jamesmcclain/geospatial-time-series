#!/usr/bin/env python3

# BSD 3-Clause License
#
# Copyright (c) 2022-23
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
import json
import logging
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from pytorch_metric_learning import losses
from torch.utils.data import DataLoader

from autoencoders import MultiViewAutoencoder
from datasets import SeriesDataset, SeriesEmbedDataset
from models import (SeriesEfficientNetb0, SeriesMobileNetv3, SeriesResNet18,
                    SeriesResNet34, SeriesResNet50)


def remove_empty_text_rows(a, b):
    mask = (a[:, 0] < torch.inf)
    return a[mask], b[mask]


if __name__ == "__main__":

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # yapf: disable
    # Command line arguments
    parser = argparse.ArgumentParser(description="Pretrain a model using a bunch unlabeled Sentinel-2 time series")
    parser.add_argument("cog_dirs", nargs="+", type=str, help="Paths to the data")
    parser.add_argument("--architecture", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "mobilenetv3", "efficientnetb0"], help="The model architecture to use (default: resnet18)")
    parser.add_argument("--bands", type=int, nargs="+", default=list(range(1, 12 + 1)), help="The Sentinel-2 bands to use (1 indexed)")
    parser.add_argument("--batch-size", type=int, default=7, help="The batch size (default: 7)")
    parser.add_argument("--dataset", type=str, required=True, choices=["embed-series", "series", "digest"], help="The type of data found in the data directories")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="The device to use for training (default: cuda)")
    parser.add_argument("--epochs", type=int, default=8, help="The number of epochs (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (default: 1e-4)")
    parser.add_argument('--pretrained', type=str2bool, default=False, help='Whether to start from pretrained weights (default: False)')
    parser.add_argument("--num-workers", type=int, default=3, help="Number of worker processes for the DataLoader (default: 3)")
    parser.add_argument("--output-dir", type=str, default=".", help="The directory where logs and artifacts will be deposited (default: .)")
    parser.add_argument("--pth-in", type=str, help="Optional path to a .pth file to use as a starting point for model training")
    parser.add_argument("--pth-out", type=str, default="model.pth", help="The name of the output .pth file (default: model.pth)")
    parser.add_argument("--series-length", type=int, default=8, help="The number of time steps in each sample (default: 8)")
    parser.add_argument("--latent-dims", type=int, default=8, help="The number of shared latent dimensions (default: 8)")
    parser.add_argument("--size", type=int, default=512, help="The tile size (default: 512)")

    # https://sagemaker.readthedocs.io/en/stable/overview.html#prepare-a-training-script
    parser.add_argument('--sm-hps', type=json.loads, default=os.environ.get('SM_HPS', None))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', None))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', None))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST', None))
    # yapf: enable
    args = parser.parse_args()

    # Dataset
    if args.dataset == "series":
        assert args.size % 64 == 0
        dataset = SeriesDataset(
            args.cog_dirs,
            size=args.size,
            series_length=args.series_length,
            bands=args.bands,
        )
    elif args.dataset == "embed-series":
        assert args.size % 64 == 0
        dataset = SeriesEmbedDataset(
            args.cog_dirs,
            size=args.size,
            series_length=args.series_length,
            bands=args.bands,
        )
    elif args.dataset == "digest":
        dataset = DigestDataset(args.cog_dirs)

    # DataLoader
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
            model = SeriesResNet18(pretrained=args.pretrained,
                                   channels=len(args.bands)).to(device)
        elif args.architecture == "resnet34":
            model = SeriesResNet34(pretrained=args.pretrained,
                                   channels=len(args.bands)).to(device)
        elif args.architecture == "resnet50":
            model = SeriesResNet50(pretrained=args.pretrained,
                                   channels=len(args.bands)).to(device)
        elif args.architecture == "mobilenetv3":
            model = SeriesMobileNetv3(pretrained=args.pretrained,
                                      channels=len(args.bands)).to(device)
        elif args.architecture == "efficientnetb0":
            model = SeriesEfficientNetb0(pretrained=args.pretrained,
                                         channels=len(args.bands)).to(device)
    elif args.pth_in:
        model = torch.load(args.pth_in, map_location=device).to(device)
        log.info(f"Model weights loaded from {args.pth_in}")

    # Autoencoder
    E = 768  # Instructor-XL embedding dimensions
    # LATENT = 8  # Dimensions of common latent space
    autoencoder = MultiViewAutoencoder(
        model.embedding_dim,
        E,
        args.latent_dims,
    ).to(device)  # yapf: disable

    # Loss functions, optimizers, schedulers
    base_obj = losses.TripletMarginLoss().to(device)
    obj1 = losses.SelfSupervisedLoss(base_obj).to(device)
    opt1 = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched1 = torch.optim.lr_scheduler.OneCycleLR(
        opt1,
        max_lr=args.lr,
        steps_per_epoch=len(dataloader),
        epochs=args.epochs,
    )
    scaler1 = torch.cuda.amp.GradScaler()
    if args.dataset == "embed-series":
        obj2 = torch.nn.MSELoss().to(device)
        params = list(model.parameters()) + list(autoencoder.parameters())  # yapf: disable
        opt2 = torch.optim.Adam(params, lr=args.lr)
        sched2 = torch.optim.lr_scheduler.OneCycleLR(
            opt2,
            max_lr=args.lr,
            steps_per_epoch=len(dataloader),
            epochs=args.epochs)
        scaler2 = torch.cuda.amp.GradScaler()

    for epoch in range(0, args.epochs):

        model.train()
        if args.dataset == "embed-series":
            autoencoder.train()

        triplet_losses = []
        autoencoder_losses = []

        for data in tqdm.tqdm(dataloader):

            if args.dataset == "embed-series":
                imagery0, imagery1, text_embs = data
                imagery0 = imagery0.to(device)
                imagery1 = imagery1.to(device)
                text_embs = text_embs.to(device)
            else:
                imagery0, imagery1 = data
                imagery0 = imagery0.to(device)
                imagery1 = imagery1.to(device)

            # Autoencoder
            if args.dataset == "embed-series":
                # yapf: disable
                _text_embs, _imagery = remove_empty_text_rows(text_embs, imagery0)
                if _text_embs.shape[0] > 0:
                    assert _text_embs.shape[0] == _imagery.shape[0]
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        _visual_embs = model(_imagery)
                        _visual_embs = F.normalize(_visual_embs, dim=1)
                        res = autoencoder(_visual_embs, _text_embs)
                        loss2 = 0.
                        loss2 += obj2(res[0], _visual_embs)
                        loss2 += obj2(res[1], _text_embs)
                        loss2 += 3 * obj2(res[2], res[3])
                    autoencoder_losses.append(loss2.item())
                    scaler2.scale(loss2).backward()
                    scaler2.step(opt2)
                    scaler2.update()
                    opt2.zero_grad()
                # yapf: enable

            # Body
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss1 = obj1(model(imagery0), model(imagery1))
            triplet_losses.append(loss1.item())
            scaler1.scale(loss1).backward()
            scaler1.step(opt1)
            scaler1.update()
            opt1.zero_grad()

            # # Autoencoder
            # if args.dataset == "embed-series":
            #     # yapf: disable
            #     _text_embs, _imagery = remove_empty_text_rows(text_embs, imagery1)
            #     if _text_embs.shape[0] > 0:
            #         assert _text_embs.shape[0] == _imagery.shape[0]
            #         with torch.cuda.amp.autocast(dtype=torch.float16):
            #             _visual_embs = model(_imagery)
            #             _visual_embs = F.normalize(_visual_embs, dim=1)
            #             res = autoencoder(_visual_embs, _text_embs)
            #             loss2 = 0.
            #             loss2 += obj2(res[0], _visual_embs)
            #             loss2 += obj2(res[1], _text_embs)
            #             loss2 += 3 * obj2(res[2], res[3])
            #         autoencoder_losses.append(loss2.item())
            #         loss2.backward()
            #         opt2.step()
            #         opt2.zero_grad()
            #     # yapf: enable

            # Step schedulers
            sched1.step()
            sched2.step()

        triplet_losses = np.mean(triplet_losses)
        if args.dataset != "embed-series":
            log.info(f"epoch={epoch:03} \t triplet={triplet_losses:1.5f}")
        elif args.dataset == "embed-series":
            autoencoder_losses = np.mean(autoencoder_losses)
            log.info(f"epoch={epoch:03} \t triplet={triplet_losses:1.5f} \t autoencoder={autoencoder_losses:1.5f}")  # yapf: disable

        if args.output_dir is not None and epoch < (args.epochs - 1):
            model_save_path = f"{args.output_dir}/{args.pth_out}"
            autoencoder_save_path = model_save_path.replace(".pth", f"-autoencoder-{epoch}.pth")  # yapf: disable
            torch.save(autoencoder, autoencoder_save_path)
            model_save_path = model_save_path.replace(".pth", f"-{epoch}.pth")  # yapf: disable
            torch.save(model, model_save_path)

    # Save the model after the last epoch if output_dir is provided
    if args.output_dir is not None:
        model_save_path = f"{args.output_dir}/{args.pth_out}"
        autoencoder_save_path = model_save_path.replace(".pth", f"-autoencoder.pth")  # yapf: disable
        torch.save(autoencoder, autoencoder_save_path)
        log.info(f"Autoencoder saved to {autoencoder_save_path}")
        torch.save(model, model_save_path)
        log.info(f"Model saved to {model_save_path}")
