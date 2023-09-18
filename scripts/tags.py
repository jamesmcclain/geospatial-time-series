#!/usr/bin/env python3

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
    except:
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

    log.info(f"loading relations from Daylight Map Distribution from {args.daylight_relations}")  # yapf: disable
    parquet_files = glob.glob(f"{args.daylight_relations}/*", recursive=True)  # yapf: disable
    df = pd.read_parquet(parquet_files)

    all_tags = {}

    log.info(f"querying extents from {args.extent_dir}")
    for filename in tqdm.tqdm(glob.glob(f"{args.extent_dir}/**/*.extent.npz", recursive=True)):  # yapf: disable
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
