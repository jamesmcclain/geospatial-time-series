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

import pandas as pd
import tqdm


def is_farmland(tags_dict):
    farmland = ("landuse", "farmland") in tags_dict.items()
    crop = "crop" in tags_dict.keys()
    return farmland and crop


def is_orchard(tags_dict):
    return ("landuse", "orchard") in tags_dict.items()


def is_meadow(tags_dict):
    return ("landuse", "meadow") in tags_dict.items()


def is_vineyard(tags_dict):
    return ("landuse", "vineyard") in tags_dict.items()


def is_residential(tags_dict):
    return ("landuse", "residential") in tags_dict.items()


def is_industrial(tags_dict):
    return ("landuse", "industrial") in tags_dict.items()


def is_grass(tags_dict):
    return ("landuse", "grass") in tags_dict.items()


def farmland_score(tags_dict):
    if tags_dict is None:
        return 0
    score = 0
    if is_farmland(tags_dict):
        score += 1
    if is_orchard(tags_dict):
        score -= 3
    if is_meadow(tags_dict):
        score -= 3
    if is_vineyard(tags_dict):
        score -= 3
    if is_residential(tags_dict):
        score -= 3
    if is_industrial(tags_dict):
        score -= 3
    if is_grass(tags_dict):
        score -= 3
    return score


def orchard_score(tags_dict):
    if tags_dict is None:
        return 0
    score = 0
    if is_farmland(tags_dict):
        score -= 3
    if is_orchard(tags_dict):
        score += 1
    if is_meadow(tags_dict):
        score -= 3
    if is_vineyard(tags_dict):
        score -= 3
    if is_residential(tags_dict):
        score -= 3
    if is_industrial(tags_dict):
        score -= 3
    if is_grass(tags_dict):
        score -= 3
    return score


def meadow_score(tags_dict):
    if tags_dict is None:
        return 0
    score = 0
    if is_farmland(tags_dict):
        score -= 3
    if is_orchard(tags_dict):
        score -= 3
    if is_meadow(tags_dict):
        score += 1
    if is_vineyard(tags_dict):
        score -= 3
    if is_residential(tags_dict):
        score -= 3
    if is_industrial(tags_dict):
        score -= 3
    if is_grass(tags_dict):
        score -= 3
    return score


def vineyard_score(tags_dict):
    if tags_dict is None:
        return 0
    score = 0
    if is_farmland(tags_dict):
        score -= 3
    if is_orchard(tags_dict):
        score -= 3
    if is_meadow(tags_dict):
        score -= 3
    if is_vineyard(tags_dict):
        score += 1
    if is_residential(tags_dict):
        score -= 3
    if is_industrial(tags_dict):
        score -= 3
    if is_grass(tags_dict):
        score -= 3
    return score


def residential_score(tags_dict):
    if tags_dict is None:
        return 0
    score = 0
    if is_farmland(tags_dict):
        score -= 3
    if is_orchard(tags_dict):
        score -= 3
    if is_meadow(tags_dict):
        score -= 3
    if is_vineyard(tags_dict):
        score -= 3
    if is_residential(tags_dict):
        score += 1
    if is_industrial(tags_dict):
        score -= 3
    if is_grass(tags_dict):
        score -= 3
    return score


def industrial_score(tags_dict):
    if tags_dict is None:
        return 0
    score = 0
    if is_farmland(tags_dict):
        score -= 3
    if is_orchard(tags_dict):
        score -= 3
    if is_meadow(tags_dict):
        score -= 3
    if is_vineyard(tags_dict):
        score -= 3
    if is_residential(tags_dict):
        score -= 3
    if is_industrial(tags_dict):
        score += 1
    if is_grass(tags_dict):
        score -= 3
    return score


def grass_score(tags_dict):
    if tags_dict is None:
        return 0
    score = 0
    if is_farmland(tags_dict):
        score -= 3
    if is_orchard(tags_dict):
        score -= 3
    if is_meadow(tags_dict):
        score -= 3
    if is_vineyard(tags_dict):
        score -= 3
    if is_residential(tags_dict):
        score -= 3
    if is_industrial(tags_dict):
        score -= 3
    if is_grass(tags_dict):
        score += 1
    return score


if __name__ == "__main__":
    logging.basicConfig()
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    # yapf: disable
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--daylight-relations", type=str, required=True, help="Directory containing Daylight Map Distribution relations parquet files")
    parser.add_argument("--k", type=int, required=False, default=3, help="How many top k quadkeys to fetch")
    parser.add_argument("--quadkey-json", required=False, default="quadkey.json", help="Where to store the discovered quadkeys")
    parser.add_argument("--remove-levels", required=False, type=int, default=4, help="The number of quadkey levels to remove")
    args = parser.parse_args()
    # yapf: enable

    log.info(
        f"loading relations from Daylight Map Distribution from {args.daylight_relations}"
    )
    parquet_files = glob.glob(f"{args.daylight_relations}/*", recursive=True)
    df = pd.read_parquet(parquet_files)

    df["quadkey_11"] = df["quadkey"].str[: -args.remove_levels]

    key_fn_pairs = [
        ("farmland_score", farmland_score),
        ("orchard_score", orchard_score),
        ("meadow_score", meadow_score),
        ("vineyard_score", vineyard_score),
        ("residential_score", residential_score),
        ("industrial_score", industrial_score),
        ("grass_score", grass_score),
    ]

    output = {}

    for key, fn in tqdm.tqdm(key_fn_pairs):
        df[key] = df["tags"].apply(fn)
        output.update({key: []})
        for quadkey_11, score in (
            df.groupby("quadkey_11")[key]
            .sum()
            .sort_values(ascending=False)
            .head(args.k)
            .items()
        ):
            output.get(key).append([quadkey_11, score])
        df.drop(columns=[key], inplace=True)

        with open(args.quadkey_json, "w") as json_file:
            json.dump(output, json_file, indent=4)
