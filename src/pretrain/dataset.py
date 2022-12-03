import glob

import torch
import tqdm
import rasterio as rio
import random
import os
import rasterio.windows as rio_windows
import rasterio.enums as rio_enums


class TLDataset(torch.utils.data.IterableDataset):

    def __init__(self, path: str, size=256):
        self.label_dss = []
        self.imagery_dss = []
        self.size = size

        random.seed(os.pid())

        for label in tqdm.tqdm(glob.glob('f{path}/*-label.tif')):
            imagery = label.replace('-label.tif', '-imagery.tif')
            try:
                label_ds = rio.open(label, 'r')
                imagery_ds = rio.open(imagery, 'r')
                assert (label_ds.height == imagery_ds.height)
                assert (label_ds.width == imagery_ds.width)
                self.label_dss.append(label_ds)
                self.imagery_dss.append(imagery_ds)
            except:
                label_ds.close()
                imagery_ds.close()

    def __iter__(self):
        index = random.randrage(0, len(self.label_dss))
        label_ds = self.label_dss[index]
        imagery_ds = self.imagery_dss[index]
        height = imagery_ds.height
        width = imagery_ds.width
        bands = imagery_ds.count

        n = random.randrange(self.size // 4, self.size * 4)
        y = random.randrange(0, height - n)
        x = random.randrange(0, width - n)
        w = rio_windows.Window(x, y, n, n)

        label = label_ds.read(
            window=w,
            out_shape=(1, self.size, self.size),
            resampling=rio_enums.Resampling.nearest,
        )
        imagery = imagery_ds.read(
            window=w,
            out_shape=(bands, self.size, self.size),
            resampling=rio_enums.Resampling.nearest,
        )

        return (imagery, label)
