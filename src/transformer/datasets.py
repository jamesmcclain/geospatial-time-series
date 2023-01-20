import glob
import math
import random
from typing import List

import numpy as np
import rasterio as rio
import torch
import tqdm


class InMemorySeasonalDataset(torch.utils.data.IterableDataset):

    def __init__(self,
                 series_paths: List[str],
                 label_path: str,
                 iters_per_incr: int,
                 size: int = 32,
                 dimensions: int = 512,
                 sequence_limit: int = 10,
                 digest_labels: bool = False,
                 evaluation: bool = False):
        self.size = size
        assert dimensions % 2 == 0
        self.dimensions = dimensions
        self.sequence_limit = min(sequence_limit, len(series_paths))
        self.digest_labels = digest_labels
        self.evaluation = evaluation

        self.data = []
        self.labels = []
        self.pos_encoding = []

        self.width = None
        self.height = None
        self.bands = None

        self.iters = 0
        assert iters_per_incr > 0
        self.iters_per_incr = iters_per_incr

        self.pos_template = np.zeros((1, self.dimensions))
        x = np.power(8, 2.0 / dimensions)
        self.pos_template[0, 0::2] = np.power(x, range(0, dimensions // 2))
        self.pos_template[0, 1::2] = np.power(x, range(0, dimensions // 2))

        if self.evaluation == False:
            desc = 'Loading training data'
        elif self.evaluation == True:
            desc = 'Loading evaluation data'

        for filename in tqdm.tqdm(series_paths, desc=desc):
            with rio.open(filename, 'r') as ds:
                if self.height is None:
                    height = self.height = ds.height
                if self.width is None:
                    width = self.width = ds.width
                if self.bands is None:
                    self.bands = ds.count
                assert self.height == ds.height
                assert self.width == ds.width
                assert self.bands == ds.count

                width80 = int(width * 0.80)
                if self.evaluation == False:
                    w = rio.windows.Window(0, 0, width80, height)
                elif self.evaluation == True:
                    w = rio.windows.Window(width80, 0, width - width80, height)

                month = int(filename.split('/')[-4])
                day = int(filename.split('/')[-3])
                day = float(day + 31 * (month - 1))
                day = 2.0 * math.pi * (day / 372)

                self.data.append(ds.read(window=w))
                pos = self.pos_template * day
                pos[:, 0::2] = np.sin(pos[:, 0::2])
                pos[:, 1::2] = np.cos(pos[:, 1::2])
                self.pos_encoding.append(pos)

        with rio.open(label_path, 'r') as ds:
            self.labels = ds.read(window=w)

        self.data = np.stack(self.data, axis=0).astype(np.float32)
        self.pos_encoding = np.concatenate(self.pos_encoding,
                                           axis=0).astype(np.float32)

    def __iter__(self):
        return self

    def __next__(self):
        size = self.size
        _, _, height, width = self.data.shape

        # A random window
        x = random.randrange(0, width - size)
        y = random.randrange(0, height - size)

        imagery = self.data[:, :, y:(y + size), x:(x + size)]
        labels = self.labels[:, y:(y + size), x:(x + size)].astype(np.uint8)
        if self.digest_labels:
            labels = np.array([
                np.mean(labels == 1),  # farms
                # np.mean(labels == 2),  # forests
                # np.mean(labels == 3),  # roads
            ]).astype(np.float32)

        # Sample from a subset of the mosaics
        ss, _, _, _ = imagery.shape
        offset = self.iters // self.iters_per_incr
        indxs = np.arange(0, self.sequence_limit, dtype=np.uint32)
        indxs = (indxs + offset) % ss
        imagery = imagery[indxs]
        pos_encoding = self.pos_encoding[indxs]

        self.iters = self.iters + 1
        return (imagery, labels, pos_encoding)


class RawSeriesDataset(torch.utils.data.IterableDataset):

    def __init__(self,
                 series_paths: List[str],
                 target_path: str,
                 size: int = 256,
                 max_seq=20,
                 evaluation: bool = False,
                 channels=None):
        self.series = []
        self.target = None
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

        with rio.open(target_path, 'r') as ds:
            assert self.height == ds.height
            assert self.width == ds.width
            assert self.bands == ds.count
            self.target = target_path

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
        with rio.open(self.target, 'r') as ds:
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
