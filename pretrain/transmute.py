#!/usr/bin/env python3

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

import argparse

import torch

if __name__ == "__main__":
    # yapf: disable
    # Command line arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--pth-in", type=str, required=True, help="Path to a .pth file to use for input")
    parser.add_argument("--onnx-out", type=str, required=True, help="Where to save the .onnx output file")
    # yapf: enable
    args = parser.parse_args()

    torch.hub.load(
        "jamesmcclain/CheapLab:21a2cd4cfe02ca48c7fdd58f47e121236c09e657",
        "make_cheaplab_model",
        num_channels=12,
        out_channels=3,
        preshrink=2,
        trust_repo=True,
        skip_validation=True,
    ),

    device = torch.device("cpu")
    model = torch.load(args.pth_in, map_location=device).to(device)
    model.eval()

    x = torch.randn(1, 12, 512, 512)

    torch.onnx.export(
        model,
        x,
        args.onnx_out,
        verbose=True,
        input_names=["data"],
        output_names=["output"],
        dynamic_axes={
            "data": {0: "batch_size", 2: "x_dim", 3: "y_dim"},
            "output": {0: "batch_size"},
        },
    )
