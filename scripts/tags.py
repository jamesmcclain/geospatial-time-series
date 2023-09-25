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
import glob
import json
import logging

import numpy as np
import pandas as pd
import tqdm
from pyquadkey2 import quadkey


def dict_or_none(stuff):
    try:
        return dict(stuff)
    except Exception:
        return dict()


def tags_to_dict(tags):
    tags = dict_or_none(tags)
    return {key: value for key, value in tags.items() if value is not None}


def extent_query(extent, df):
    lat = (extent[1] + extent[3]) / 2
    lon = (extent[0] + extent[2]) / 2
    qk = quadkey.from_geo((lat, lon), args.bing_tile_level)
    re = f"^{qk}"

    df = df[df["quadkey"].str.contains(re)]
    df = df[
        (df["min_lat"] <= max(extent[1], extent[3]))
        & (df["max_lat"] >= min(extent[1], extent[3]))
        & (df["min_lon"] <= max(extent[0], extent[2]))
        & (df["max_lon"] >= max(extent[0], extent[2]))
    ]

    return df


if __name__ == "__main__":
    logging.basicConfig()
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    # yapf: disable
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--daylight-relations", type=str, required=True, help="Directory containing Daylight Map Distribution relations parquet files")
    parser.add_argument("--extent-dir", type=str, required=True, help="Directory containing .npz files that contain extents")
    parser.add_argument("--bing-tile-level", type=int, default=5, help="The Bing tile level at-which to query Daylight")
    parser.add_argument("--tags", type=str, default="tags.json", help="Where to save the extracted tags")
    args = parser.parse_args()
    # yapf: enable

    log.info(
        f"loading relations from Daylight Map Distribution from {args.daylight_relations}"
    )
    parquet_files = glob.glob(f"{args.daylight_relations}/*", recursive=True)
    df = pd.read_parquet(parquet_files)

    all_tags = {}

    log.info(f"querying extents from {args.extent_dir}")
    for filename in tqdm.tqdm(
        glob.glob(f"{args.extent_dir}/**/*.extent.npz", recursive=True)
    ):
        base_filename = filename.split("/")[-1]
        base_filename = base_filename.split(".")[0]
        extent = np.load(filename).get("extent")
        filt = extent_query(extent, df)
        filt["tags"] = filt["tags"].apply(tags_to_dict)
        all_tags[base_filename] = {}
        for tags in filt["tags"]:
            all_tags[base_filename].update(tags)

    with open(args.tags, "w") as file:
        json.dump(all_tags, file, indent=4)
