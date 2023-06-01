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
                 dump_mode: bool = False):
        super().__init__()
        self.dim = dim
        self.dataset_length = 0
        self.nuggets = []
        self.debug = debug
        self.dump_mode = dump_mode

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

        nugget_index = index - nugget_start_index  # Index within nugget

        # Get the group
        group_index_a = (nugget_index + 0) % len(groups)
        group_a = groups[group_index_a]
        if not self.dump_mode:
            group_index_b = (nugget_index + 1) % len(groups)
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
        for filename_a in group_a:
            with rio.open(filename_a, "r") as ds:
                imagery_a.append(ds.read(window=w).astype(np.float32))
        imagery_a = torch.from_numpy(np.stack(imagery_a, axis=0))

        if not self.dump_mode:
            imagery_b = []
            for filename_b in group_b:
                with rio.open(filename_b, "r") as ds:
                    imagery_b.append(ds.read(window=w).astype(np.float32))
            imagery_b = torch.from_numpy(np.stack(imagery_b, axis=0))

        # Return imagery
        if not self.dump_mode:
            return (imagery_a, imagery_b)
        else:
            return (imagery_a, current_nugget, group_index_a, tile_y, tile_x)
