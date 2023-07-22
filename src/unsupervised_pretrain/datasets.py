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
from typing import List

import numpy as np
import pyproj
import rasterio as rio
import torch


def split_list(lst, chunk_size):
    sublists = []
    for i in range(0, len(lst), chunk_size):
        sublists.append(lst[i:i + chunk_size])
    return sublists


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
                 size: int = 512,
                 series_length: int = 5,
                 text_mode: bool = False):
        super().__init__()
        self.size = size
        self.dataset_length = 0
        self.nuggets = []
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
            blocks_tall = height // size
            blocks_wide = width // size
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

    def nugget_and_groups(self, index):

        # Get the nugget
        nugget_start_index = 0
        for current_nugget, nugget in enumerate(self.nuggets):
            nugget_length = nugget.get("nugget_length")
            if index - nugget_start_index < nugget_length:
                break
            nugget_start_index += nugget_length

        groups = nugget.get("groups")

        nugget_relative_index = index - nugget_start_index  # Index within nugget

        # Get the groups
        group_index_a = (nugget_relative_index + 0) % len(groups)
        group_index_b = (nugget_relative_index + 1) % len(groups)

        return (nugget, group_index_a, group_index_b, nugget_relative_index)

    def __getitem__(self, index):
        if index >= self.dataset_length:
            raise StopIteration()

        nugget, group_index_a, group_index_b, nugget_relative_index = self.nugget_and_groups(index)

        groups = nugget.get("groups")
        blocks_tall = nugget.get("blocks_tall")
        blocks_wide = nugget.get("blocks_wide")

        group_a = groups[group_index_a]
        group_b = groups[group_index_b]

        # Calculate the window
        tile_relative_index = nugget_relative_index // len(groups)  # yapf:disable Index within tile
        tile_y = tile_relative_index % blocks_wide
        tile_x = tile_relative_index // blocks_wide
        # yapf: disable
        w = rio.windows.Window(
            tile_x * self.size,
            tile_y * self.size,
            self.size,
            self.size,
        )
        # yapf: disable

        # Return text
        if self.text_mode:
            return (w, nugget)

        # Read the imagery
        imagery_a = []
        for filename_a in group_a:
            with rio.open(filename_a, "r") as ds:
                imagery_a.append(ds.read(window=w).astype(np.float32))
        imagery_a = torch.from_numpy(np.stack(imagery_a, axis=0))

        if not self.text_mode:
            imagery_b = []
            for filename_b in group_b:
                with rio.open(filename_b, "r") as ds:
                    imagery_b.append(ds.read(window=w).astype(np.float32))
            imagery_b = torch.from_numpy(np.stack(imagery_b, axis=0))

        # Return imagery
        if not self.text_mode:
            return (imagery_a, imagery_b)
        else:
            raise NotImplemented()

class SeriesEmbedDataset(SeriesDataset):

    def __init__(self,
                 cog_dirs: List[str],
                 size: int = 512,
                 series_length: int = 5):

        super().__init__(
            cog_dirs = cog_dirs,
            size = size,
            series_length = series_length
        )

        for cog_dir, nugget in zip(cog_dirs, self.nuggets):
            cog_dir_parts = cog_dir.split("/")
            while len(cog_dir_parts[-1]) == 0:
                cog_dir_parts = cog_dir_parts[:-1]
            embedding_filename = f"{cog_dir_parts[-1]}-{size}-{series_length}.npy"
            embedding_filename = f"{cog_dir}/**/{embedding_filename}"
            embedding_filename = glob.glob(embedding_filename, recursive=True)[-1]
            nugget["embeddings"] = np.load(embedding_filename)


    def __getitem__(self, index):

        nugget, group_index_a, group_index_b, _ = self.nugget_and_groups(index)
        imagery_a, imagery_b = super().__getitem__(index)
        embedding_a = nugget.get("embeddings")[group_index_a]
        embedding_b = nugget.get("embeddings")[group_index_b]

        return (imagery_a, imagery_b, embedding_a, embedding_b)
