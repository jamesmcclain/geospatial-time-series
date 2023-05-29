import glob
from typing import List

import numpy as np
import rasterio as rio
import torch


def split_list(lst, chunk_size):
    sublists = []
    for i in range(0, len(lst), chunk_size):
        sublists.append(lst[i:i + chunk_size])
    return sublists


class SeriesDataset(torch.utils.data.Dataset):

    def __init__(self,
                 cog_dirs: List[str],
                 dim: int = 512,
                 series_length: int = 5,
                 debug: bool = False):
        super().__init__()
        self.dim = dim
        self.dataset_length = 0
        self.nuggets = []
        self.debug = debug

        for cog_dir in cog_dirs:
            cog_list = glob.glob(f"{cog_dir}/**/cog.tif", recursive=True)
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
        nugget_start_index = 0
        for nugget in self.nuggets:
            nugget_length = nugget.get("nugget_length")
            if index - nugget_start_index < nugget_length:
                break
            nugget_start_index += nugget_length

        groups = nugget.get("groups")
        blocks_tall = nugget.get("blocks_tall")
        blocks_wide = nugget.get("blocks_wide")

        nugget_index = index - nugget_start_index  # Index within nugget

        # Get the group
        group_index_a = (nugget_index + 0) % len(groups)
        group_index_b = (nugget_index + 1) % len(groups)
        group_a = groups[group_index_a]
        group_b = groups[group_index_b]

        # Calculate the window
        tile_index = nugget_index // len(groups)  # Index within tile
        tile_y = tile_index % blocks_wide
        tile_x = tile_index // blocks_wide
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

        # Read the imagery
        imagery_a = []
        imagery_b = []
        for filename_a, filename_b in zip(group_a, group_b):
            with rio.open(filename_a, "r") as ds:
                imagery_a.append(ds.read(window=w).astype(np.float32))
            with rio.open(filename_b, "r") as ds:
                imagery_b.append(ds.read(window=w).astype(np.float32))
        imagery_a = np.stack(imagery_a, axis=0)
        imagery_b = np.stack(imagery_b, axis=0)
        return (imagery_a, imagery_b)
