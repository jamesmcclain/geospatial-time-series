# BSD 3-Clause License
#
# Copyright (c) 2022-23
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
import random
import threading
from typing import List
from zipfile import BadZipFile

import numpy as np
import torch


class SeriesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir: str,
        series_length: int = 10,
        bands=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ):
        super().__init__()
        self.series_length = series_length
        self.bands = copy.copy(bands)

        self.chips = sorted(glob.glob(f"{dataset_dir}/**/*.chip.npz", recursive=True))
        self.locks = []
        for chip in self.chips:
            self.locks.append(threading.Lock())
        # self.lock = threading.Lock()

    def __len__(self):
        return len(self.chips)

    def __getitem__(self, index):
        # with self.locks[index]:
        if True:
            data = np.load(self.chips[index])
            chip = np.array(data.get("chips"))
        n, _, _, _ = chip.shape
        assert n >= 2 * self.series_length
        perm = list(range(0, n))
        random.shuffle(perm)

        left = perm[0 : self.series_length]
        left = chip[left, ...][:, self.bands, ...].astype(np.float32)

        right = perm[self.series_length : 2 * self.series_length]
        right = chip[right, ...][:, self.bands, ...].astype(np.float32)

        return left, right, index


class SeriesEmbedDataset(SeriesDataset):
    def __init__(
        self,
        dataset_dir: str,
        embeddings_npz: str,
        series_length: int = 10,
        bands=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ):
        super().__init__(
            dataset_dir=dataset_dir,
            series_length=series_length,
            bands=bands,
        )

        self.embeddings = {}
        for k, v in np.load(embeddings_npz).items():
            self.embeddings.update({k: v})

    def __getitem__(self, index):
        left, right = super().__getitem__(index)
        chip_id = self.chips[index].split("/")[-1].split(".")[0]
        embedding = self.embeddings.get(chip_id)

        return left, right, embedding, index
