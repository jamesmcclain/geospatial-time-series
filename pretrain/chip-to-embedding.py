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
import logging
import sys
import tarfile

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from datasets import SeriesDataset, SeriesEmbedDataset

if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--autocast", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="The autocast type (default: bfloat16)")
    parser.add_argument("--bands", type=int, nargs="+", default=list(range(0, 12)), help="The Sentinel-2 bands to use (0 indexed)")
    parser.add_argument("--batch-size", type=int, default=8, help="The batch size (default: 8)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="The device to use for training (default: cuda)")
    parser.add_argument("--input-dir", type=str, required=True, help="Where to find the input data")
    parser.add_argument("--num-workers", type=int, default=3, help="Number of worker processes for the DataLoader (default: 3)")
    parser.add_argument("--output-npz", type=str, required=True, help="Where to put the embeddings")
    parser.add_argument("--sequence", type=int, nargs="+", default=list(range(2, 31, 2)), help="The sequence of time steps (default: 2 to 30, inclusive, step size 2)")
    parser.add_argument("--tarball-in", type=str, required=True, help="Path to a tarball containing the .pth files")
    args = parser.parse_args()
    # yapf: enable

    # Logging
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format="%(asctime)-15s %(message)s"
    )
    log = logging.getLogger()

    # PyTorch device
    device = torch.device(args.device)

    # Model
    log.info(f"Model weights loaded from {args.tarball_in}")
    with tarfile.open(args.tarball_in, "r:gz") as tar:
        model = torch.load(tar.extractfile("model.pth"), map_location=device).to(device)
        model.eval()
        autoencoder = torch.load(
            tar.extractfile("model-autoencoder.pth"), map_location=device
        ).to(device)
        autoencoder.eval()

    # Prepare dictionary of results
    results = {}
    for i in args.sequence:
        results.update({str(i): []})

    dtype = eval(f"torch.{args.autocast}")
    for i in tqdm.tqdm(args.sequence):
        # Dataset and dataloader
        dataset = SeriesDataset(args.input_dir, i // 2, args.bands)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
            prefetch_factor=1,
        )

        # Inference
        with torch.inference_mode():
            for left, right, _ in tqdm.tqdm(dataloader, leave=False):
                with torch.cuda.amp.autocast(dtype=dtype):
                    # if True:
                    data = torch.cat([left, right], dim=1).to(device)
                    # result = autoencoder.autoencoder_1.encoder(model(data)).cpu().numpy()
                    result = model(data).cpu().numpy()
                    results.get(str(i)).append(result)

        results[str(i)] = np.concatenate(results.get(str(i)), axis=0)

    np.savez_compressed(args.output_npz, **results)
