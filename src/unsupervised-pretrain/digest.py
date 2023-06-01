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

import torch
import tqdm

from datasets import SeriesDataset

if __name__ == "__main__":
    # yapf: disable
    # Command line arguments
    parser = argparse.ArgumentParser(description="Pretrain a model using a bunch unlabeled Sentinel-2 time series")
    parser.add_argument("cog_dirs", nargs="+", type=str, help="Path to the training set tarball")
    parser.add_argument("--output-dir", type=str, default=".", help="The directory where logs and artifacts will be deposited (default: .)")
    # yapf: enable
    args = parser.parse_args()

    dim = 512
    series_length = 5
    dataset = SeriesDataset(args.cog_dirs,
                            dim=dim,
                            series_length=series_length,
                            dump_mode=True)

    for t in tqdm.tqdm(dataset):
        (imagery, nugget, group, y, x) = t
        torch.save(imagery, f"{args.output_dir}/{nugget}-{group}-{y}-{x}.pt")
