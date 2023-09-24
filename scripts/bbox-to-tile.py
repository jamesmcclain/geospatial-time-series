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
import bz2
import gzip
import json
import lzma

import tqdm
from shapely.geometry import shape
from shapely.strtree import STRtree


def load_geojson(filename):
    if filename.endswith(".xz"):
        with lzma.open(filename, "rt") as file:
            return json.load(file)
    elif filename.endswith(".bz2"):
        with bz2.open(filename, "rt") as file:
            return json.load(file)
    elif filename.endswith(".gz"):
        with gzip.open(filename, "rt") as file:
            return json.load(file)
    else:
        with open(filename, "r") as file:
            return json.load(file)


def save_geojson(data, filename):
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)


def build_spatial_index(type2_features):
    polygons = [shape(feature["geometry"]) for feature in type2_features]
    return STRtree(polygons)


def find_largest_intersection(type1_polygon, type2_features, spatial_index):
    largest_intersection_area = 0
    largest_intersection_name = None

    # Use spatial index to find candidates for intersection
    possible_matches = spatial_index.query(type1_polygon)
    possible_matches = spatial_index.geometries.take(possible_matches).tolist()
    for type2_polygon in possible_matches:
        intersection = type1_polygon.intersection(type2_polygon)
        if intersection.is_empty:
            continue
        intersection_area = intersection.area
        if intersection_area > largest_intersection_area:
            # Find the corresponding feature
            index = next(
                i
                for i, feature in enumerate(type2_features)
                if shape(feature["geometry"]) == type2_polygon
            )
            largest_intersection_area = intersection_area
            largest_intersection_name = type2_features[index]["properties"]["Name"]

    return largest_intersection_name


def main():
    # yapf: disable
    parser = argparse.ArgumentParser(description="Find largest intersections between two GeoJSON files.")
    parser.add_argument("--type1", nargs="+", required=True, help="Filenames for the type 1 GeoJSON files.")
    parser.add_argument("--type2", required=True, help="Filename for the type 2 GeoJSON file.")
    parser.add_argument("--output", nargs="+", required=True, help="Filenames for the output GeoJSON files.")
    args = parser.parse_args()
    # yapf: enable

    # The type1 file comes from quad-to-bbox.py
    # The type2 file comes from here: https://github.com/justinelliotmeyers/Sentinel-2-Shapefile-Index
    type2_data = load_geojson(args.type2)
    type2_features = type2_data["features"]
    spatial_index = build_spatial_index(type2_features)

    for type1, output in tqdm.tqdm(list(zip(args.type1, args.output))):
        type1_data = load_geojson(type1)
        for feature in type1_data["features"]:
            type1_polygon = shape(feature["geometry"])
            largest_intersection_name = find_largest_intersection(
                type1_polygon, type2_data["features"], spatial_index
            )
            if largest_intersection_name:
                feature["properties"]["IntersectedWith"] = largest_intersection_name

        save_geojson(type1_data, output)


if __name__ == "__main__":
    main()
