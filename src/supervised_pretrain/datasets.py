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
import math
import random

import numpy as np
import rasterio as rio
import torch


class TLDataset(torch.utils.data.IterableDataset):

    def __init__(self, path: str, size=256, evaluation=False):
        self.pairs = []
        self.size = size
        self.evaluation = evaluation

        for label_filename in glob.glob(f'{path}/*-label.tif'):
            imagery_filename = label_filename.replace('-label.tif',
                                                      '-imagery.tif')
            try:
                with rio.open(label_filename, 'r') as label_ds, rio.open(
                        imagery_filename, 'r') as imagery_ds:
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

        with rio.open(label_filename,
                      'r') as label_ds, rio.open(imagery_filename,
                                                 'r') as imagery_ds:
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
                    y = random.randrange(int(math.sqrt(0.80) * height),
                                         height - n)
                    x = random.randrange(0, width - n)
                else:
                    _n = int((1.0 - math.sqrt(0.80)) * width)
                    n = random.randrange(_n // 2, _n)
                    y = random.randrange(0, height - n)
                    x = random.randrange(int(math.sqrt(0.80) * width),
                                         width - n)

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
        label[imagery[0, :, :] == 0] == 0
        return (imagery, label)
