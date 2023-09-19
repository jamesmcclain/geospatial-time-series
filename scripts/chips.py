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
import json
import logging
import random
import uuid

import boto3
import numpy as np
import rasterio
import rasterio as rio
from pyproj import CRS
from rasterio.warp import transform
from rasterio.windows import Window

INDEX_TO_NAME = {
    0: "B01.tif",
    1: "B02.tif",
    2: "B03.tif",
    3: "B04.tif",
    4: "B05.tif",
    5: "B06.tif",
    6: "B07.tif",
    7: "B08.tif",
    8: "B8A.tif",
    9: "B09.tif",
    10: "B11.tif",
    11: "B12.tif",
}


def random_element_generator(data):
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    for element in shuffled_data:
        yield element


def list_redband_files(bucket, prefix):
    s3 = boto3.client('s3')
    tif_files = []

    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('B04.tif'):
                tif_files.append(obj['Key'])

    return tif_files


def pixel_to_wgs84_rect(ds, pixel_bounds):
    """
    Convert a pixel-space rectangle to a WGS84 rectangle.

    Args:
    - ds: A rasterio dataset.
    - pixel_bounds: A tuple of (left, bottom, right, top) in pixel coordinates.

    Returns:
    - A tuple of (left, bottom, right, top) in WGS84 coordinates.
    """
    # Convert pixel bounds to dataset CRS using affine transformation
    left, top = ds.transform * (pixel_bounds[0], pixel_bounds[3])
    right, bottom = ds.transform * (pixel_bounds[2], pixel_bounds[1])

    # Convert the dataset CRS bounds to WGS84
    dst_crs = CRS.from_epsg(4326)  # WGS84
    xs = [left, right]
    ys = [top, bottom]
    lon, lat = transform(ds.crs, dst_crs, xs, ys)

    return (lon[0], lat[1], lon[1], lat[0])  # Reordering to match WGS84 format


def wgs84_to_pixel_rect(ds, wgs84_bounds):
    """
    Convert a WGS84 rectangle to a pixel-space rectangle.

    Args:
    - ds: A rasterio dataset.
    - wgs84_bounds: A tuple of (left, bottom, right, top) in WGS84 coordinates.

    Returns:
    - A tuple of (left, bottom, right, top) in pixel coordinates.
    """
    # Convert WGS84 bounds to dataset CRS
    dst_crs = CRS.from_epsg(4326)  # WGS84
    xs = [wgs84_bounds[0], wgs84_bounds[2]]
    ys = [wgs84_bounds[1], wgs84_bounds[3]]
    lon, lat = transform(dst_crs, ds.crs, xs, ys)

    # Convert dataset CRS bounds to pixel-space using the inverse affine transformation
    left_pixel, top_pixel = ~ds.transform * (lon[0], lat[1])
    right_pixel, bottom_pixel = ~ds.transform * (lon[1], lat[0])

    left_pixel = int(left_pixel)
    right_pixel = int(right_pixel)
    top_pixel = int(top_pixel)
    bottom_pixel = int(bottom_pixel)

    return (left_pixel, bottom_pixel, right_pixel, top_pixel)


def write_bbox_to_geojson(wgs84_bounds, filename):
    """
    Write a WGS84 bounding box to a GeoJSON file.

    Args:
    - wgs84_bounds: A tuple of (left, bottom, right, top) in WGS84 coordinates.
    - filename: Name of the GeoJSON file to write to.

    Returns:
    - None
    """
    # Create the GeoJSON structure
    geojson_data = {
        "type":
        "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type":
                "Polygon",
                "coordinates": [[
                    [wgs84_bounds[0], wgs84_bounds[1]],  # Bottom-left
                    [wgs84_bounds[0], wgs84_bounds[3]],  # Top-left
                    [wgs84_bounds[2], wgs84_bounds[3]],  # Top-right
                    [wgs84_bounds[2], wgs84_bounds[1]],  # Bottom-right
                    [wgs84_bounds[0], wgs84_bounds[1]],  # Close the loop
                ]]
            }
        }]
    }

    # Write to file
    with open(filename, 'w') as outfile:
        json.dump(geojson_data, outfile, indent=4)


def read_one_location(size, threshold, series, bucket_name, prefix):
    red_band_list = list_redband_files(bucket_name, prefix)
    log.info(f"{len(red_band_list)} candidate tiles")
    gen = random_element_generator(red_band_list)

    # Search for suitable area in the tile
    red_band = f"/vsis3/{bucket_name}/{next(gen)}"
    with rio.open(red_band, 'r') as ds:
        width = ds.width
        height = ds.height

        data = np.zeros([1, 1, 1], dtype=np.int16)
        while not (((data > 0) * (data < 5000)).sum() > threshold):
            x = random.randint(0, width - size - 1)
            y = random.randint(0, height - size - 1)
            window = Window(x, y, size, size)
            log.info(f"trying {x}, {y}")
            data = ds.read(window=window, out_shape=(1, size, size))

        wgs84_box = pixel_to_wgs84_rect(ds, [x, y, x + size, y + size])

    chips = []

    while len(chips) < series:

        log.info(f"{len(chips)} chips so far")
        bands = [None] * 12
        with rio.open(red_band, 'r') as ds:
            bands[3] = data = ds.read(window=window, out_shape=(1, size, size))
        good_data = (data > 0) * (data < 5000)

        # If this chip is good (not too much nodata and not too much
        # "cloud"), then read all 12 L2A bands
        if good_data.sum() > threshold:
            for i in range(0, 12):
                this_band = red_band.replace("B04.tif", INDEX_TO_NAME.get(i))
                with rio.open(this_band, 'r') as ds:
                    pixel_box = wgs84_to_pixel_rect(ds, wgs84_box)
                    x1, y1, x2, y2 = pixel_box
                    window = Window(x1, y1, x2 - x1, y2 - y1)
                    bands[i] = ds.read(window=window,
                                       out_shape=(1, size, size))
            chip = np.concatenate(bands, axis=0)
            chips.append(chip)

        try:
            red_band = f"/vsis3/{bucket_name}/{next(gen)}"
            log.info(red_band)
        except StopIteration:
            return

    if len(chips) >= series:
        chips = np.stack(chips, axis=0)
        return chips, wgs84_box
    else:
        return


if __name__ == "__main__":

    logging.basicConfig()
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    # yapf: disable
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--series", type=int, required=False, default=1, help="The number of chips to collect")
    parser.add_argument("--good-data-threshold", type=float, required=False, default=0.80, help="The minimum proportion of pixels needed for a chip to be \"good\"")
    parser.add_argument("--size", type=int, required=False, default=512, help="The linear size (in pixels) of each chip")
    parser.add_argument("--output-chip-dir", type=str, required=False, default="/tmp/", help="Where to store the retrieved chips")
    parser.add_argument("--output-extent-dir", type=str, required=False, default="/tmp/", help="Where to store the extents of the retrieved chips")
    parser.add_argument("--bucket", type=str, required=False, default="sentinel-cogs", help="The S3 bucket with the source data")
    parser.add_argument("--prefix", type=str, required=False, default="sentinel-s2-l2a-cogs/11/S/KU/", help="The S3 prefix for the source data")
    args = parser.parse_args()
    # yapf: enable

    bucket_name = args.bucket
    prefix = args.prefix
    size = args.size
    series = args.series
    thr = size * size * args.good_data_threshold

    chips, wgs84_box = read_one_location(size, thr, series, bucket_name, prefix)

    log.info("writing files")
    basename = uuid.uuid4().hex
    np.savez_compressed(f"{args.output_chip_dir}/{basename}.chip", chips=chips)
    np.savez_compressed(f"{args.output_extent_dir}/{basename}.extent",
                        extent=np.array(wgs84_box, dtype=float))
