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
import glob
import json
import logging
import math
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import tqdm
from pytorch_metric_learning import losses
from pytorch_metric_learning.utils import distributed as pml_dist
from torch.utils.data import DataLoader

from autoencoders import MultiViewAutoencoder
from datasets import SeriesDataset, SeriesEmbedDataset
from models import (
    SeriesEfficientNetb0,
    SeriesMobileNetv3,
    SeriesResNet18,
    SeriesResNet34,
    SeriesResNet50,
)


class CheckGradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Save the input for backward pass
        ctx.save_for_backward(input)
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # Get the saved input
        input, = ctx.saved_tensors

        # Check if the gradient has been modified in-place
        if not torch.equal(grad_output, grad_output.clone()):
            print("Gradient modified in-place!")
            import pdb; pdb.set_trace()

        return grad_output


def check_gradient(tensor):
    return CheckGradientFunction.apply(tensor)

if __name__ == "__main__":

    def str2bool(v):
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    # yapf: disable
    parser = argparse.ArgumentParser(description="Pretrain a model using unlabeled Sentinel-2 time series")
    # I/O locations
    parser.add_argument("--input-dir", type=str, required=False, default=os.environ.get("SM_CHANNEL_TRAIN", None), help="Path to input data")
    parser.add_argument("--output-dir", type=str, required=False, default=os.environ.get("SM_MODEL_DIR", "."), help="Where the trained model will be deposited")
    parser.add_argument("--checkpoint-dir", type=str, required=False, default=None, help="Where checkpoints will be written")
    parser.add_argument("--embeddings-npz", type=str, required=False, default=None, help="Where to find the embeddings")
    # Hyperparameters
    parser.add_argument("--architecture", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "mobilenetv3", "efficientnetb0"], help="The model architecture to use (default: resnet18)")
    parser.add_argument("--autocast", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="The autocast type (default: bfloat16)")
    parser.add_argument("--bands", type=int, nargs="+", default=list(range(0, 12)), help="The Sentinel-2 bands to use (0 indexed)")
    parser.add_argument("--batch-size", type=int, default=7, help="The batch size (default: 7)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="The device to use for training (default: cuda)")
    parser.add_argument("--epochs", type=int, default=8, help="The number of epochs (default: 8)")
    parser.add_argument("--latent-dims", type=int, default=8, help="The number of shared latent dimensions (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (default: 1e-4)")
    parser.add_argument("--num-workers", type=int, default=3, help="Number of worker processes for the DataLoader (default: 3)")
    parser.add_argument("--pretrained", type=str2bool, default=False, help="Whether to start from pretrained weights (default: False)")
    parser.add_argument("--pth-in", type=str, help="Optional path to a .pth file to use as a starting point for model training")
    parser.add_argument("--pth-out", type=str, default="model.pth", help="The name of the output .pth file (default: model.pth)")
    parser.add_argument("--series-length", type=int, default=8, help="The number of time steps in each sample (default: 8)")
    # https://sagemaker.readthedocs.io/en/stable/overview.html#prepare-a-training-script
    parser.add_argument("--master-addr", type=str, default=None, help="The address of the master node (for distributed training)")
    parser.add_argument("--master-port", type=str, default=None, help="The port on the master node (for distributed training)")
    parser.add_argument("--rank", type=int, default=None, help="The rank of this node (for distributed training)")
    parser.add_argument("--world-size", type=int, default=None, help="The world size (for distributed training)")
    parser.add_argument("--sm-backend", type=str, default="gloo", choices=["gloo", "nccl"], help="The PyTorch distributed-trained backend to use")
    parser.add_argument("--sm-current-host", type=str, default=os.environ.get("SM_CURRENT_HOST", None))
    parser.add_argument("--sm-hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS", "null")))
    parser.add_argument("--sm-data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", None))
    parser.add_argument("--sm-hps", type=json.loads, default=os.environ.get("SM_HPS", "null"))
    args = parser.parse_args()
    if args.input_dir is None:
        parser.error("Must supply --input-dir")
    if (args.sm_hosts is not None and args.sm_current_host is None) or (args.sm_hosts is None and args.sm_current_host is not None):
        parser.error("If --sm-current-host is given then --sm-hosts must be given (and vice-versa)")
    if (args.rank is not None and args.world_size is None) or (args.rank is None and args.world_size is not None):
        parser.error("If --rank is given then --world-size must be given (and vice-versa)")
    if args.rank is not None and args.sm_current_host is not None:
        parser.error("Must use either --rank and --world-size or --sm-current-host or --sm-hosts, but not both")
    # yapf: enable

    # Logging
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format="%(asctime)-15s %(message)s"
    )
    log = logging.getLogger()
    if args.checkpoint_dir is not None:
        fh = logging.FileHandler(f"{args.checkpoint_dir}/output.log")
        log.addHandler(fh)
    log.info(args.__dict__)

    if args.rank is not None and args.world_size is not None:
        rank = args.rank
        world_size = args.world_size
    elif args.sm_current_host is not None and args.sm_hosts is not None:
        rank = sorted(args.sm_hosts, reverse=False).index(args.sm_current_host)
        world_size = len(args.sm_hosts)
        log.info(f"rank {rank} via inspection of hosts")
    else:
        rank = 0
        world_size = 1

    # Initialize distribution (if it should be initialized)
    if world_size > 1:
        if args.master_addr is not None:
            os.environ["MASTER_ADDR"] = args.master_addr
        if args.master_port is not None:
            os.environ["MASTER_PORT"] = args.master_port
        dist.init_process_group(
            backend=args.sm_backend, rank=rank, world_size=world_size
        )
        log.info(
            f"Distributed training: rank={rank} world_size={world_size} backend={args.sm_backend}"
        )

    # Dataset
    if args.embeddings_npz is None:
        dataset = SeriesDataset(args.input_dir, args.series_length, args.bands)
    else:
        embeddings_npz = glob.glob(
            f"{args.input_dir}/**/{args.embeddings_npz}", recursive=True
        )[0]
        dataset = SeriesEmbedDataset(
            args.input_dir, embeddings_npz, args.series_length, args.bands
        )

    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None

    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        pin_memory=True,
        num_workers=args.num_workers,
        sampler=sampler,
        prefetch_factor=1,
    )

    # PyTorch device
    device = torch.device(args.device)

    # Model
    if args.pth_in is None:
        if args.architecture == "resnet18":
            model = SeriesResNet18(
                pretrained=args.pretrained, channels=len(args.bands)
            ).to(device)
        elif args.architecture == "resnet34":
            model = SeriesResNet34(
                pretrained=args.pretrained, channels=len(args.bands)
            ).to(device)
        elif args.architecture == "resnet50":
            model = SeriesResNet50(
                pretrained=args.pretrained, channels=len(args.bands)
            ).to(device)
        elif args.architecture == "mobilenetv3":
            model = SeriesMobileNetv3(
                pretrained=args.pretrained, channels=len(args.bands)
            ).to(device)
        elif args.architecture == "efficientnetb0":
            model = SeriesEfficientNetb0(
                pretrained=args.pretrained, channels=len(args.bands)
            ).to(device)
    elif args.pth_in:
        model = torch.load(args.pth_in, map_location=device).to(device)
        log.info(f"Model weights loaded from {args.pth_in}")

    # Autoencoder
    E = 25  # GloVe embedding dimensions
    autoencoder = MultiViewAutoencoder(
        model.embedding_dim,
        E,
        args.latent_dims,
    ).to(device)

    # Make model and autoencoder distributed (if they should be)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model)
        autoencoder = torch.nn.parallel.DistributedDataParallel(autoencoder)

    # Loss functions, optimizers, schedulers
    obj1 = losses.TripletMarginLoss(triplets_per_anchor=args.batch_size).to(device)
    if world_size > 1:
        obj1 = pml_dist.DistributedLossWrapper(obj1).to(device)
        log.info("wrapping w/ DistributedLossWrapper")
    opt1 = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched1 = torch.optim.lr_scheduler.OneCycleLR(
        opt1,
        max_lr=args.lr,
        steps_per_epoch=len(dataloader),
        epochs=args.epochs,
    )
    scaler1 = torch.cuda.amp.GradScaler()

    # Again
    if args.embeddings_npz is not None:
        obj2 = torch.nn.MSELoss().to(device)
        params = list(model.parameters()) + list(autoencoder.parameters())
        opt2 = torch.optim.Adam(params, lr=args.lr)
        sched2 = torch.optim.lr_scheduler.OneCycleLR(
            opt2, max_lr=args.lr, steps_per_epoch=len(dataloader), epochs=args.epochs
        )
        scaler2 = torch.cuda.amp.GradScaler()

    model.train()
    if args.embeddings_npz is not None:
        autoencoder.train()

    torch.autograd.set_detect_anomaly(True)  # XXX
    for epoch in range(0, args.epochs):
        triplet_losses = []
        autoencoder_losses = []

        dtype = eval(f"torch.{args.autocast}")

        for data in tqdm.tqdm(dataloader):
            if len(data) == 3:
                left, right, labels = data
                left = check_gradient(left.to(device))
                right = check_gradient(right.to(device))
                labels = labels.to(device)
                embeddings = None
            elif len(data) == 4:
                left, right, embeddings, lables = data
                left = left.to(device)
                right = right.to(device)
                embeddings = embeddings.to(device)
                labels = labels.to(device)
            else:
                raise Exception(f"Cannot handle data tuple of length {len(data)}")

            # Body
            opt1.zero_grad()
            with torch.cuda.amp.autocast(dtype=dtype):
                loss1 = obj1(
                    torch.cat([model(left), model(right)], dim=0),
                    torch.cat([labels, labels], dim=0),
                )
            triplet_losses.append(loss1.item())
            scaler1.scale(loss1).backward()
            scaler1.step(opt1)
            scaler1.update()

            # Autoencoder
            if embeddings is not None:
                # yapf: disable
                opt2.zero_grad()
                with torch.cuda.amp.autocast(dtype=dtype):
                    visual_left = model(left)
                    result = autoencoder(
                        visual_left,
                        embeddings,
                    )
                    loss2 = obj2(visual_left, result[0])
                    loss2 += obj2(embeddings, result[1])
                    loss2 += obj2(result[2], result[3])
                autoencoder_losses.append(loss2.item())
                scaler2.scale(loss2).backward()
                scaler2.step(opt2)
                scaler2.update()
                # yapf: enable

            # Step schedulers
            sched1.step()
            if embeddings is not None:
                sched2.step()

        triplet_losses = np.mean(triplet_losses)
        if embeddings is None and rank == 0:
            log.info(f"epoch={epoch:03} \t triplet={triplet_losses:1.5f}")
        elif embeddings is not None and rank == 0:
            autoencoder_losses = np.mean(autoencoder_losses)
            log.info(
                f"epoch={epoch:03} \t triplet={triplet_losses:1.5f} \t autoencoder={autoencoder_losses:1.5f}"
            )

        if args.checkpoint_dir is not None and epoch < (args.epochs - 1) and rank == 0:
            model_save_path = f"{args.checkpoint_dir}/{args.pth_out}"
            if embeddings is not None:
                autoencoder_save_path = model_save_path.replace(
                    ".pth", f"-autoencoder-{epoch}.pth"
                )
                torch.save(autoencoder, autoencoder_save_path)
            model_save_path = model_save_path.replace(".pth", f"-{epoch}.pth")
            torch.save(model, model_save_path)

    # Save the model after the last epoch if output_dir is provided
    if args.output_dir is not None and rank == 0:
        model_save_path = f"{args.output_dir}/{args.pth_out}"
        autoencoder_save_path = model_save_path.replace(".pth", f"-autoencoder.pth")
        if embeddings is not None:
            torch.save(autoencoder, autoencoder_save_path)
            log.info(f"Autoencoder saved to {autoencoder_save_path}")
        torch.save(model, model_save_path)
        log.info(f"Model saved to {model_save_path}")
