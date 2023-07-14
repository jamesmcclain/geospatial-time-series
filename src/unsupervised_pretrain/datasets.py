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

import glob
import random
from typing import List

import numpy as np
import pyproj
import rasterio as rio
import torch
from pyproj import CRS, Transformer
from rasterio.transform import Affine
from shapely.geometry import Point, box
from shapely.ops import unary_union
from shapely.wkt import loads
import math


def split_list(lst, chunk_size):
    sublists = []
    for i in range(0, len(lst), chunk_size):
        sublists.append(lst[i:i + chunk_size])
    return sublists


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


def dict_to_string(input_dict):
    pairs = []
    for key, value in input_dict.items():
        pair = f"{key}: {value}"
        pairs.append(pair)
    return ", ".join(pairs)


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
            if percent < 1.:
                continue  # Suppress areas that are less than 1% of the scene
            if "natural" in tags or "landuse" in tags:
                lulc = tags.get("natural", tags.get("landuse"))
                line = f"There is {lulc} area that occupies {percent:.1f}% of the visible area."
            elif "leisure" in tags:
                leisure = tags.get("leisure").replace("_", " ")
                line = f"There is a {leisure} (leisure area) that occupies {percent:.1f}% of the visible area."
            else:
                line = (
                    f"There is an area that occupies {percent:.1f}% "
                    f"of the visible area that has tags: \"{tags}\"."
                )
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
        plural_noun = "a plethora"

    building_types = ", ".join(building_types)

    if len(rows) > 0:
        random.shuffle(lines)
        first_line = f"There are {plural_noun} buildings in the scene"
        if len(building_types) > 0:
            first_line += f" of type {building_types}."
        else:
            first_line += "."
        # # x0, y0 = bbox.exterior.coords[0]
        # # x1, y1 = bbox.exterior.coords[2]
        # # first_line = (
        # #     f"There are {label_count} labels, "
        # #     f"of which {building_count} are buildings and "
        # #     f"{label_count - building_count} are non-buildings. "
        # #     f"Building labels occupy {building_pct:.1f}% of the visible area, "
        # #     f"while non-building labels occupy {nonbuilding_pct:.1f}% of the visible area. "
        # #     "The bounding box of the visible area has corners at "
        # #     f"latitude {min(y0, y1)} and longitude {min(x0, x1)}, and "
        # #     f"latitude {max(y0, y1)} and longitude {max(x0, x1)}\n")
        # # first_line = (
        # #     f"The scene has {building_count} buildings. "
        # #     f"Labeled areas occupy {nonbuilding_pct:.1f}% of the visible area.")
        # first_line = f"The scene has {building_count} buildings. "
        lines = [first_line] + lines
    else:
        lines = ["No information is available about this area."]

    return " ".join(lines)


class DigestDataset(torch.utils.data.Dataset):

    def __init__(self, pt_dirs: List[str]):
        self.pt_list = []
        for pt_dir in pt_dirs:
            pt_list = glob.glob(f"{pt_dir}/**/*.pt", recursive=True)
            self.pt_list += pt_list

    def __len__(self):
        return len(self.pt_list)

    def __getitem__(self, index):
        if index >= len(self.pt_list):
            raise StopIteration()

        this_pt = self.pt_list[index]
        this_data = torch.load(this_pt)

        pt_dir, pt_filename = this_pt.rsplit("/", 1)
        nugget, group, y, x_rest = pt_filename.split("-")
        group = int(group)
        try:
            that_data = torch.load(f"{pt_dir}/{nugget}-{group+1}-{y}-{x_rest}")
        except:
            that_data = torch.load(f"{pt_dir}/{nugget}-0-{y}-{x_rest}")

        return (this_data, that_data)


class SeriesDataset(torch.utils.data.Dataset):

    def __init__(self,
                 cog_dirs: List[str],
                 dim: int = 512,
                 series_length: int = 5,
                 debug: bool = False,
                 dump_mode: bool = False,
                 text_mode: bool = False):
        super().__init__()
        self.dim = dim
        self.dataset_length = 0
        self.nuggets = []
        self.debug = debug
        self.dump_mode = dump_mode
        self.text_mode = text_mode

        for cog_dir in sorted(cog_dirs):
            cog_list = sorted(glob.glob(f"{cog_dir}/**/cog.tif", recursive=True))  # yapf: disable
            cog_list.sort()
            with rio.open(cog_list[0], "r") as ds:
                height = ds.height
                width = ds.width
            for cog in cog_list[1:]:
                with rio.open(cog, "r") as ds:
                    assert height == ds.height
                    assert width == ds.width
            rest = series_length - (len(cog_list) % series_length)
            if rest != 0:
                cog_list = cog_list + cog_list[:rest]
            assert len(cog_list) % series_length == 0
            groups = split_list(cog_list, series_length)
            blocks_tall = height // dim
            blocks_wide = width // dim
            nugget_length = blocks_tall * blocks_wide * len(groups)
            nugget = {
                "groups": groups,
                "blocks_tall": blocks_tall,
                "blocks_wide": blocks_wide,
                "nugget_length": nugget_length,
            }
            self.nuggets.append(nugget)
            self.dataset_length += nugget_length

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        if index >= self.dataset_length:
            raise StopIteration()

        nugget_start_index = 0
        for current_nugget, nugget in enumerate(self.nuggets):
            nugget_length = nugget.get("nugget_length")
            if index - nugget_start_index < nugget_length:
                break
            nugget_start_index += nugget_length

        groups = nugget.get("groups")
        blocks_tall = nugget.get("blocks_tall")
        blocks_wide = nugget.get("blocks_wide")

        nugget_relative_index = index - nugget_start_index  # Index within nugget

        # Get the group
        group_index_a = (nugget_relative_index + 0) % len(groups)
        group_a = groups[group_index_a]
        if not self.dump_mode:
            group_index_b = (nugget_relative_index + 1) % len(groups)
            group_b = groups[group_index_b]

        # Calculate the window
        tile_relative_index = nugget_relative_index // len(groups)  # Index within tile
        tile_y = tile_relative_index % blocks_wide
        tile_x = tile_relative_index // blocks_wide
        # yapf: disable
        w = rio.windows.Window(
            tile_x * self.dim,
            tile_y * self.dim,
            self.dim,
            self.dim,
        )
        # yapf: disable

        if self.debug:
            print(group_a[0], group_b[0], tile_x, tile_y)

        # Return text
        if self.text_mode:
            return (w, nugget)

        # Read the imagery
        imagery_a = []
        for filename_a in group_a:
            with rio.open(filename_a, "r") as ds:
                imagery_a.append(ds.read(window=w).astype(np.float32))
        imagery_a = torch.from_numpy(np.stack(imagery_a, axis=0))

        if not self.dump_mode and not self.text_mode:
            imagery_b = []
            for filename_b in group_b:
                with rio.open(filename_b, "r") as ds:
                    imagery_b.append(ds.read(window=w).astype(np.float32))
            imagery_b = torch.from_numpy(np.stack(imagery_b, axis=0))

        # Return imagery
        if not self.dump_mode and not self.text_mode:
            return (imagery_a, imagery_b)
        elif self.dump_mode:
            return (imagery_a, current_nugget, group_index_a, tile_y, tile_x)
        else:
            raise NotImplemented()


class SeriesParquetDataset(SeriesDataset):

    def __init__(self,
                 cog_dirs: List[str],
                 dim: int = 512,
                 series_length: int = 5,
                 debug: bool = False,
                 dump_mode: bool = False):

        super().__init__(
            cog_dirs = cog_dirs,
            dim = dim,
            series_length = series_length,
            debug = debug,
            dump_mode = False,
            text_mode = True)

        import geopandas as gpd
        import pandas as pd

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
            parquet_files = glob.glob(f"{cog_dir}/**/*.parquet", recursive=True)
            dfs = []
            for parquet_file in parquet_files:
                df = pd.read_parquet(parquet_files)[["wkt", "tags"]]
                df["geometry"] = df["wkt"].apply(loads)
                df["tags"] = df["tags"].apply(dict_or_none)
                dfs.append(df)
            df = pd.concat(dfs)
            gdf = gpd.GeoDataFrame(df)

            cog = self.nuggets[idx].get("groups")[0][0]
            pixel_to_wgs84 = create_pixel_to_wgs84_transformer(cog)
            self.nuggets[idx].update({"gdf": gdf, "pixel_to_wgs84": pixel_to_wgs84})

    def __getitem__(self, index):
        w, nugget = super().__getitem__(index)

        pixel_to_wgs84 = nugget.get("pixel_to_wgs84")
        lon_min, lat_min = pixel_to_wgs84(w.col_off, w.row_off)
        lon_max, lat_max = pixel_to_wgs84(w.col_off + w.width, w.row_off + w.height)
        bbox = box(lon_min, lat_min, lon_max, lat_max)
        gdf = nugget.get("gdf")
        intersections = gdf[gdf.intersects(bbox)].copy()
        intersections["clipped_geometry"] = intersections["geometry"].intersection(bbox)

        return rows_to_text(intersections, bbox)
