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

from pyquadkey2 import quadkey


def quadkey_to_geojson_polygon(qk: str) -> dict:
    # Get the tile from the quadkey
    tile = quadkey.from_str(qk)

    # Get the bounding box of the tile
    bbox_nw = tile.to_geo(quadkey.TileAnchor.ANCHOR_NW)
    bbox_se = tile.to_geo(quadkey.TileAnchor.ANCHOR_SE)

    geojson_polygon = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [bbox_nw[1], bbox_nw[0]],
                    [bbox_se[1], bbox_nw[0]],
                    [bbox_se[1], bbox_se[0]],
                    [bbox_nw[1], bbox_se[0]],
                    [bbox_nw[1], bbox_nw[0]],
                ]
            ],
        },
        "properties": {"quadkey": qk},
    }

    return geojson_polygon


def main():
    # yapf: disable
    parser = argparse.ArgumentParser(description="Convert Bing Map Tile quadkeys to GeoJSON bounding boxes.")
    parser.add_argument("--input-file", required=True, help="Path to the input JSON file containing quadkeys.")
    parser.add_argument("--output-file", required=True, help="Path to the output GeoJSON file.")
    parser.add_argument("--key", required=True, help="The key within the json file to use.")
    args = parser.parse_args()
    # yapf: enable

    with open(args.input_file, "r") as f:
        data = json.load(f)

    quadkeys = [item[0] for item in data[args.key]]

    features = [quadkey_to_geojson_polygon(qk) for qk in quadkeys]

    geojson_collection = {"type": "FeatureCollection", "features": features}

    with open(args.output_file, "w") as f:
        return json.dump(geojson_collection, f, indent=4)


if __name__ == "__main__":
    main()
