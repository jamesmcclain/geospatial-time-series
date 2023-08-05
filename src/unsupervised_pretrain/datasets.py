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

import copy
import glob
from typing import List

import numpy as np
import rasterio as rio
import torch
from pyproj import Proj


def split_list(lst, chunk_size):
    sublists = []
    for i in range(0, len(lst), chunk_size):
        sublists.append(lst[i:i + chunk_size])
    return sublists


class SeriesDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        cog_dirs: List[str],
        size: int = 512,
        series_length: int = 5,
        text_mode: bool = False,
    ):
        super().__init__()
        self.size = size
        self.dataset_length = 0
        self.cog_dirs = []
        self.text_mode = text_mode

        for cog_dir in sorted(cog_dirs):

            # Get entire file list
            cog_list = sorted(glob.glob(f"{cog_dir}/**/cog.tif", recursive=True))  # yapf: disable
            cog_list.sort()

            # Check that all files have the same dimensions
            with rio.open(cog_list[0], "r") as ds:
                height = ds.height
                width = ds.width
                crs = ds.crs
            for cog in cog_list[1:]:
                with rio.open(cog, "r") as ds:
                    assert height == ds.height
                    assert width == ds.width
                    assert crs == ds.crs

            # Make the file list length a multipel of the series length
            rest = series_length - (len(cog_list) % series_length)
            if rest != 0:
                cog_list = cog_list + cog_list[:rest]
            assert len(cog_list) % series_length == 0

            # Geometry
            groups = split_list(cog_list, series_length)
            blocks_tall = height // size
            blocks_wide = width // size
            cog_dir_length = blocks_tall * blocks_wide * len(groups)

            # Orientation
            src_proj = Proj(crs)
            latlon = src_proj.crs.axis_info[0].direction.lower() in {"north", "south"}  # yapf: disable

            _cog_dir = {
                "blocks_tall": blocks_tall,
                "blocks_wide": blocks_wide,
                "groups": groups,
                "latlon": latlon,
                "cog_dir_length": cog_dir_length,
            }
            self.cog_dirs.append(copy.copy(_cog_dir))
            self.dataset_length += cog_dir_length

    def __len__(self):
        return self.dataset_length

    def cog_dir_and_groups(self, index):

        # Get the cog_dir
        cog_dir_start_index = 0
        for current_cog_dir, cog_dir in enumerate(self.cog_dirs):
            cog_dir_length = cog_dir.get("cog_dir_length")
            if index - cog_dir_start_index < cog_dir_length:
                break
            cog_dir_start_index += cog_dir_length

        cog_dir_relative_index = index - cog_dir_start_index  # Index within cog_dir

        return (cog_dir, cog_dir_relative_index)

    def __getitem__(self, index):
        if index >= self.dataset_length:
            raise StopIteration()

        cog_dir, cog_dir_relative_index = self.cog_dir_and_groups(index)  # yapf:disable
        groups = cog_dir.get("groups")
        group_index_a = (cog_dir_relative_index + 0) % len(groups)
        group_index_b = (cog_dir_relative_index + 1) % len(groups)
        blocks_tall = cog_dir.get("blocks_tall")
        blocks_wide = cog_dir.get("blocks_wide")

        group_a = groups[group_index_a]
        group_b = groups[group_index_b]

        # Calculate the window
        group_relative_index = cog_dir_relative_index // len(groups)  # yapf:disable Index within the group
        group_y = group_relative_index % blocks_wide
        group_x = group_relative_index // blocks_wide
        # yapf: disable
        if cog_dir.get("latlon"):
            w = rio.windows.Window(
                group_y * self.size,
                group_x * self.size,
                self.size,
                self.size,
            )
        else:
            w = rio.windows.Window(
                group_x * self.size,
                group_y * self.size,
                self.size,
                self.size,
            )
        # yapf: disable

        # Return text
        if self.text_mode:
            return (w, cog_dir)

        # Read the imagery
        imagery_a = []
        for filename_a in group_a:
            with rio.open(filename_a, "r") as ds:
                imagery_a.append(ds.read(window=w).astype(np.float32))
        imagery_a = torch.from_numpy(np.stack(imagery_a, axis=0))

        imagery_b = []
        for filename_b in group_b:
            with rio.open(filename_b, "r") as ds:
                imagery_b.append(ds.read(window=w).astype(np.float32))
        imagery_b = torch.from_numpy(np.stack(imagery_b, axis=0))

        assert imagery_a.shape[2] != 0 and imagery_b.shape[2] != 0

        return (imagery_a, imagery_b)


class SeriesEmbedDataset(SeriesDataset):

    def __init__(self,
                 cog_dirs: List[str],
                 size: int = 512,
                 series_length: int = 5,
                 ):

        cog_dirs = sorted(cog_dirs)

        super().__init__(
            cog_dirs = cog_dirs,
            size = size,
            series_length = series_length,
        )

        for cog_dir, _cog_dir in zip(cog_dirs, self.cog_dirs):
            cog_dir_parts = cog_dir.split("/")
            while len(cog_dir_parts[-1]) == 0:
                cog_dir_parts = cog_dir_parts[:-1]
            embedding_filename = f"{cog_dir_parts[-1]}-{size}.npy"
            embedding_filename = f"{cog_dir}/**/{embedding_filename}"
            embedding_filename = glob.glob(embedding_filename, recursive=True)[-1]

            vectors = np.load(embedding_filename)
            _cog_dir["embeddings"] = vectors

            blocks_tall = _cog_dir.get("blocks_tall")
            blocks_wide = _cog_dir.get("blocks_wide")
            blocks = blocks_tall * blocks_wide
            assert blocks == _cog_dir.get("embeddings").shape[0]


    def __getitem__(self, index):

        cog_dir, cog_dir_relative_index = self.cog_dir_and_groups(index)
        imagery_a, imagery_b = super().__getitem__(index)
        len_groups = len(cog_dir.get("groups"))
        group_relative_index = cog_dir_relative_index // len_groups
        embedding = cog_dir.get("embeddings")[group_relative_index]

        return (imagery_a, imagery_b, embedding)
