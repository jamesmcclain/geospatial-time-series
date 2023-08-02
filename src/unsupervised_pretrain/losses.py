# BSD 3-Clause License
#
# Copyright (c) 2023
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

import torch


class InnerProductMatchLoss(torch.nn.Module):

    def __init__(self):
        super(InnerProductMatchLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, p1, p2, y):
        ground_truth = y @ y.t()
        proj1 = p1 @ p1.t()
        proj2 = p2 @ p2.t()
        part0 = self.mse_loss(proj1, ground_truth)
        part1 = self.mse_loss(proj2, ground_truth)
        part2 = self.mse_loss(proj1, proj2)
        return part0 + part1 + (2 * part2)


class OrthogonalLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, x, y):
        assert x.shape[0] == y.shape[0]

        result = x @ y.t()
        eye = torch.eye(result.shape[0],
                        dtype=result.dtype,
                        device=result.device)

        return self.mse(result, eye)


class MaximumMeanDiscrepancyLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def gaussian_kernel(self, a, b):
        square_dist = (a.unsqueeze(0) - b.unsqueeze(1)).pow(2).sum(dim=2)
        kernel_val = (-square_dist).exp()
        return kernel_val

    def forward(self, x, y):
        assert x.shape[0] == y.shape[0]

        kernel_x = self.gaussian_kernel(x, x)
        kernel_y = self.gaussian_kernel(y, y)
        kernel_xy = self.gaussian_kernel(x, y)

        kernel_x_mean = torch.mean(kernel_x)
        kernel_y_mean = torch.mean(kernel_y)
        kernel_xy_mean = torch.mean(kernel_xy)

        return kernel_x_mean - (2 * kernel_xy_mean) + kernel_y_mean


class ComboLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.orthogonal = OrthogonalLoss()
        self.mmd = MaximumMeanDiscrepancyLoss()

    def forward(self, x, y):
        return self.orthogonal(x, y) + self.mmd(x, y)
