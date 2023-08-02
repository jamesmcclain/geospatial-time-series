#!/usr/bin/env python3

# BSD 3-Clause License
#
# Copyright (c) 2023
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
import sys

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from pytorch_metric_learning import losses
from torch.utils.data import DataLoader

from datasets import SeriesEmbedDataset

if __name__ == "__main__":
    # yapf: disable
    # Command line arguments
    parser = argparse.ArgumentParser(description="Pretrain a model using a bunch unlabeled Sentinel-2 time series")
    parser.add_argument("pth_in", type=str, help="The name of the model to use for visual embeddings")
    parser.add_argument("visual_embeddings", type=str, help="Where to store the visual embeddings")
    parser.add_argument("text_embeddings", type=str, help="Where to store the text embeddings")
    parser.add_argument("cog_dirs", nargs="+", type=str, help="Paths to the data")
    parser.add_argument("--batch-size", type=int, default=16, help="The batch size (default: 16)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="The device to use for training (default: cuda)")
    parser.add_argument("--num-workers", type=int, default=3, help="Number of worker processes for the DataLoader (default: 3)")
    parser.add_argument("--series-length", type=int, default=8, help="The number of time steps in each sample (default: 8)")
    parser.add_argument("--size", type=int, default=512, help="The tile size (default: 512)")
    # yapf: enable
    args = parser.parse_args()

    assert args.size % 64 == 0
    dataset = SeriesEmbedDataset(
        args.cog_dirs,
        size=args.size,
        series_length=args.series_length,
        dump_mode=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    # Logging
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)-15s %(message)s")  # yapf: disable
    log = logging.getLogger()
    log.info(args.__dict__)

    # PyTorch device
    device = torch.device(args.device)

    # Load model
    model = torch.load(args.pth_in, map_location=device).to(device)

    # Loop over the data
    visual_embeddings = []
    text_embeddings = []
    for data in tqdm.tqdm(dataloader):
        imagery, text_embedding = data
        text_embeddings.append(text_embedding.cpu().numpy())
        with torch.inference_mode():
            visual_embedding = model(imagery.to(device)).detach().cpu().numpy()
        visual_embeddings.append(visual_embedding)

    visual_embeddings = np.concatenate(visual_embeddings, axis=0)
    text_embeddings = np.concatenate(text_embeddings, axis=0)
    np.save(args.visual_embeddings, visual_embeddings)
    np.save(args.text_embeddings, text_embeddings)
