import glob
import math
import random
from typing import List

import numpy as np
import rasterio as rio
import torch
import tqdm


class RawSeriesDataset(torch.utils.data.IterableDataset):

    def __init__(self,
                 series_paths: List[str],
                 mosaic_path: str,
                 size: int = 256,
                 max_seq=20,
                 evaluation: bool = False,
                 channels=None):
        self.series = []
        self.mosaic = None
        self.size = size
        self.channels = channels
        self.max_seq = max_seq
        self.evaluation = evaluation

        self.height = None
        self.width = None
        self.bands = None

        for filename in series_paths:
            with rio.open(filename, 'r') as ds:
                if self.height is None:
                    self.height = ds.height
                if self.width is None:
                    self.width = ds.width
                if self.bands is None:
                    self.bands = ds.count
                assert self.height == ds.height
                assert self.width == ds.width
                assert self.bands == ds.count
                self.series.append(filename)

        with rio.open(mosaic_path, 'r') as ds:
            assert self.height == ds.height
            assert self.width == ds.width
            assert self.bands == ds.count
            self.mosaic = mosaic_path

        # self.bands = list(range(1, self.bands+1))

    def __iter__(self):
        return self

    def __next__(self):
        height = self.height
        width = self.width
        max_seq = self.max_seq

        if self.evaluation == False:
            if self.size < 64:
                n = self.size
            else:
                n = random.randrange(self.size // 2, self.size * 2)
            y = random.randrange(0, int(math.sqrt(0.80) * height) - n)
            x = random.randrange(0, int(math.sqrt(0.80) * width) - n)
        elif self.evaluation == True:
            if random.randint(0, 1) > 0:
                if self.size < 64:
                    n = self.size
                else:
                    _n = int((1.0 - math.sqrt(0.80)) * height)
                    n = random.randrange(_n // 2, _n)
                y = random.randrange(int(math.sqrt(0.80) * height), height - n)
                x = random.randrange(0, width - n)
            else:
                if self.size < 64:
                    n = self.size
                else:
                    _n = int((1.0 - math.sqrt(0.80)) * width)
                    n = random.randrange(_n // 2, _n)
                y = random.randrange(0, height - n)
                x = random.randrange(int(math.sqrt(0.80) * width), width - n)

        w = rio.windows.Window(x, y, n, n)

        # Read target
        with rio.open(self.mosaic, 'r') as ds:
            if self.channels is None:
                target = ds.read(
                    window=w,
                    out_shape=(self.bands, self.size, self.size),
                    resampling=rio.enums.Resampling.nearest,
                ).astype(np.float32)
            else:
                target = ds.read(
                    self.channels,
                    window=w,
                    out_shape=(len(self.channels), self.size, self.size),
                    resampling=rio.enums.Resampling.nearest,
                ).astype(np.float32)

        # Read source sequence
        source = []
        for filename in random.sample(self.series, len(self.series))[:max_seq]:
            with rio.open(filename, 'r') as ds:
                if self.channels is None:
                    source.append(
                        ds.read(
                            window=w,
                            out_shape=(self.bands, self.size, self.size),
                            resampling=rio.enums.Resampling.nearest,
                        ).astype(np.float32))
                else:
                    source.append(
                        ds.read(
                            self.channels,
                            window=w,
                            out_shape=(len(self.channels), self.size,
                                       self.size),
                            resampling=rio.enums.Resampling.nearest,
                        ).astype(np.float32))

        source = np.stack(source, axis=0)

        return (source, target)


class NpzSeriesDataset(torch.utils.data.Dataset):

    def __init__(self, path, narrow: bool = False, tiles: bool = False):
        assert (not narrow or not tiles)
        self.narrow = narrow
        self.tiles = tiles
        self.filenames = []
        for filename in tqdm.tqdm(glob.glob(f'{path}/*.npz')):
            try:
                _ = np.load(filename)
                self.filenames.append(filename)
            except:
                pass

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        try:
            thing = np.load(self.filenames[index])
        except:
            print(self.filenames[index])
            return None

        source = np.squeeze(thing.get('source'))
        target = np.squeeze(thing.get('target'))

        if self.narrow:
            source = np.max(source, axis=(2, 3))
            target = np.max(target, axis=(1, 2))
        if self.tiles:
            n, _, _ = source.shape
            source = source.reshape(n, -1)
            source[source > 2000.0] = 2000.0
            source = (source / 2000.0)
            target = target.reshape(-1)
            target[target > 2000.0] = 2000.0
            target = (target / 2000.0)

        return (source, target)
