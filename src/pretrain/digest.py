#!/usr/bin/env python3

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
