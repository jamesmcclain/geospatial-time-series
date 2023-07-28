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

import glob
import math
from typing import List

import geopandas as gpd
import pandas as pd
import rasterio as rio
from pyproj import CRS, Transformer
from rasterio.transform import Affine
from shapely.geometry import Point, box
from shapely.ops import unary_union
from shapely.wkt import loads

from datasets import SeriesDataset


def dict_or_none(stuff):
    try:
        return dict(stuff)
    except:
        return dict()


def remove_none_values(input_dict):
    return {
        key: value
        for key, value in input_dict.items() if value is not None
    }


def rows_to_text(rows, bbox):

    label_count = len(rows)
    building_count = 0
    total_area = bbox.area
    building_union = []
    nonbuilding_union = []

    building_types = set()

    lines = []
    for _, row in rows.iterrows():
        tags = remove_none_values(row["tags"])
        del tags["type"]
        is_building = "building" in tags
        if is_building:
            building_count = building_count + 1
            geometry = row["geometry"]
            clipped_geometry = row["clipped_geometry"]
            building_union.append(geometry)
            building_type = tags.get("building", "yes")
            if building_type.lower() != "yes":
                building_types.add(building_type)
        else:
            clipped_geometry = row["clipped_geometry"]
            percent = 100. * clipped_geometry.area / total_area
            nonbuilding_union.append(clipped_geometry)
            if percent < 5.:
                continue  # Suppress areas that are less than 5% of the scene
            if "natural" in tags or "landuse" in tags:
                lulc = tags.get("natural", tags.get("landuse"))
                line = f"There is {lulc} area that occupies {percent:.1f}% of the visible area."
            elif "leisure" in tags:
                leisure = tags.get("leisure").replace("_", " ")
                line = f"There is a {leisure} that occupies {percent:.1f}% of the visible area."
            elif "boundary" in tags:
                boundary = tags.get("boundary").replace("_", " ")
                line = f"There is {boundary} that occupies {percent:.1f}% of the visible area."
            else:
                line = f"There is a strange area that occupies {percent:.1f}% of the visible area."
            # else:
            #     line = (f"There is an area that occupies {percent:.1f}% "
            #             f"of the visible area that has tags: \"{tags}\".")
            lines.append(line)

    building_pct = 100. * unary_union(building_union).area / total_area
    nonbuilding_pct = 100. * unary_union(nonbuilding_union).area / total_area

    if building_count == 0:
        plural_noun = "zero"
    elif math.log(building_count) <= 1.:
        plural_noun = "a handful of"
    elif math.log(building_count) <= 2.:
        plural_noun = "a few"
    elif math.log(building_count) <= 3.:
        plural_noun = "many"
    elif math.log(building_count) <= 4.:
        plural_noun = "numerous"
    else:
        plural_noun = "a plethora of"

    building_types = ", ".join(building_types)

    if len(rows) > 0:
        first_line = f"There are {plural_noun} buildings in the scene"
        if len(building_types) > 0:
            first_line += f" of type {building_types}."
        else:
            first_line += "."
        lines = [first_line] + lines
    else:
        lines = ["No information is available about this area."]

    return " ".join(lines)


class ParquetHackDataset(SeriesDataset):

    def __init__(
        self,
        cog_dirs: List[str],
        size: int = 512,
    ):

        super().__init__(
            cog_dirs=cog_dirs,
            size=size,
            text_mode=True,
        )

        def create_pixel_to_wgs84_transformer(geotiff_file):
            with rio.open(geotiff_file) as src:
                transform = src.transform
                crs = src.crs

            pixel_transform = Affine(*transform[:6])
            target_crs = CRS.from_epsg(4326)  # EPSG code for WGS84
            transformer = Transformer.from_crs(crs, target_crs, always_xy=True)

            def pixel_to_wgs84(x, y):
                x_native, y_native = pixel_transform * (x, y)
                lon, lat = transformer.transform(x_native, y_native)
                return lon, lat

            return pixel_to_wgs84

        for idx, cog_dir in enumerate(cog_dirs):
            parquet_files = glob.glob(f"{cog_dir}/**/*.parquet", recursive=True)  # yapf: disable
            dfs = []
            for parquet_file in parquet_files:
                df = pd.read_parquet(parquet_files)[["wkt", "tags"]]
                df["geometry"] = df["wkt"].apply(loads)
                df["tags"] = df["tags"].apply(dict_or_none)
                dfs.append(df)
            df = pd.concat(dfs)
            gdf = gpd.GeoDataFrame(df)

            cog = self.cog_dirs[idx].get("groups")[0][0]
            pixel_to_wgs84 = create_pixel_to_wgs84_transformer(cog)
            self.cog_dirs[idx].update({
                "gdf": gdf,
                "pixel_to_wgs84": pixel_to_wgs84
            })

        # These datasets are only associated with a single tile
        self.cog_dirs = self.cog_dirs[0:1]  # One nugget
        self.cog_dirs[0]["groups"] = self.cog_dirs[0].get("groups")[0:1]  # yapf: disable One group
        self.cog_dirs[0]["groups"][0] = self.cog_dirs[0].get("groups")[0][0:1]  # yapf: disable One tile
        blocks_tall = self.cog_dirs[0].get("blocks_tall")
        blocks_wide = self.cog_dirs[0].get("blocks_wide")
        self.dataset_length = blocks_tall * blocks_wide

    def __getitem__(self, index):
        w, nugget = super().__getitem__(index)

        pixel_to_wgs84 = nugget.get("pixel_to_wgs84")
        lon_min, lat_min = pixel_to_wgs84(w.col_off, w.row_off)
        lon_max, lat_max = pixel_to_wgs84(w.col_off + w.width, w.row_off + w.height)  # yapf: disable
        bbox = box(lon_min, lat_min, lon_max, lat_max)
        gdf = nugget.get("gdf")
        intersections = gdf[gdf.intersects(bbox)].copy()
        intersections["clipped_geometry"] = intersections["geometry"].intersection(bbox)  # yapf: disable

        return rows_to_text(intersections, bbox)
