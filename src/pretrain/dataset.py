import glob

import torch
import rasterio as rio
import random
import numpy as np
import math


class TLDataset(torch.utils.data.IterableDataset):

    def __init__(self, path: str, size=256, evaluation=False):
        self.pairs = []
        self.size = size
        self.evaluation = evaluation

        for label_filename in glob.glob(f'{path}/*-label.tif'):
            imagery_filename = label_filename.replace('-label.tif', '-imagery.tif')
            try:
                with rio.open(label_filename, 'r') as label_ds, rio.open(imagery_filename, 'r') as imagery_ds:
                    assert label_ds.height == imagery_ds.height
                    assert label_ds.width == imagery_ds.width
                    self.pairs.append((imagery_filename, label_filename))
            except:
                pass

    def __iter__(self):
        return self

    def __next__(self):
        index = random.randrange(0, len(self.pairs))
        imagery_filename, label_filename = self.pairs[index]

        with rio.open(label_filename, 'r') as label_ds, rio.open(imagery_filename, 'r') as imagery_ds:
            height = imagery_ds.height
            width = imagery_ds.width
            bands = imagery_ds.count

            if self.evaluation == False:
                n = random.randrange(self.size // 4, self.size * 4)
                y = random.randrange(0, int(math.sqrt(0.80) * height) - n)
                x = random.randrange(0, int(math.sqrt(0.80) * width) - n)
            elif self.evaluation == True:
                if random.randint(0, 1) > 0:
                    _n = int((1.0 - math.sqrt(0.80)) * height)
                    n = random.randrange(_n // 2, _n)
                    y = random.randrange(int(math.sqrt(0.80) * height), height - n)
                    x = random.randrange(0, width - n)
                else:
                    _n = int((1.0 - math.sqrt(0.80)) * width)
                    n = random.randrange(_n // 2, _n)
                    y = random.randrange(0, height - n)
                    x = random.randrange(int(math.sqrt(0.80) * width), width - n)

            w = rio.windows.Window(x, y, n, n)

            try:
                label = label_ds.read(
                    window=w,
                    out_shape=(1, self.size, self.size),
                    resampling=rio.enums.Resampling.nearest,
                )
                imagery = imagery_ds.read(
                    window=w,
                    out_shape=(bands, self.size, self.size),
                    resampling=rio.enums.Resampling.nearest,
                )
            except:
                label = None
                imagery = None

        imagery = imagery.astype(np.float32)
        label = np.squeeze(label, axis=0).astype(np.long)
        label[imagery[0,:,:] == 0] == 0
        return (imagery, label)
