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
import logging
import sys

import geopandas as gpd
import pandas as pd
import rasterio
import tqdm
from pyproj import Proj, Transformer
from rasterio.warp import calculate_default_transform, reproject
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.wkt import loads


def compute_wgs84_bounding_box(geotiff_file):
    with rasterio.open(geotiff_file) as src:
        crs = src.crs
        bounds = src.bounds

    src_proj = Proj(crs)
    target_proj = Proj("EPSG:4326")
    transformer = Transformer.from_proj(src_proj, target_proj)

    minlon, minlat = transformer.transform(bounds.left, bounds.bottom)
    maxlon, maxlat = transformer.transform(bounds.right, bounds.top)

    wgs84_bbox = box(minlat, minlon, maxlat, maxlon)

    return wgs84_bbox


def dict_or_none(stuff):
    try:
        return dict(stuff)
    except:
        return dict()


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_dir", help="Path the Daylight Map Distribution directory")
    parser.add_argument("geotiff_dirs", nargs='+', help="Path to the GeoTiff directories")
    parser.add_argument("--output_dir", type=str, required=False, default=".", help="Where to deposite the output Parquet files")
    args = parser.parse_args()
    # yapf: enable

    # yapf: disable
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)-15s %(message)s")
    log = logging.getLogger()
    # yapf: enable

    # Load daylight map distribution parquet files into dataframe
    log.info("Loading parquet files")
    parquet_files = glob.glob(f"{args.parquet_dir}/*")
    dfs = []
    for parquet_file in tqdm.tqdm(parquet_files):
        df = pd.read_parquet(parquet_file)[["wkt", "tags"]]
        df["geometry"] = df["wkt"].apply(loads)
        df["tags"] = df["tags"].apply(dict_or_none)
        dfs.append(df)
    df = pd.concat(dfs)

    log.info("DataFrame -> GeoDataFrame")
    gdf = gpd.GeoDataFrame(df)

    # Compute subsets of the dataframe
    log.info("Subsetting GeoDataFrame")
    for geotiff_dir in tqdm.tqdm(args.geotiff_dirs):

        bboxs = []
        geotiffs = glob.glob(f"{geotiff_dir}/**/cog.tif", recursive=True)
        for geotiff in geotiffs:
            bbox = compute_wgs84_bounding_box(geotiff)
            bboxs.append(bbox)

        bbox = unary_union(bbox)
        nugget = list(filter(lambda s: len(s) > 0, geotiff_dir.split("/")))[-1]

        inter = gdf[gdf.intersects(bbox)]
        inter[["wkt", "tags"]].to_parquet(f"{args.output_dir}/{nugget}.parquet")  # yapf: disable
