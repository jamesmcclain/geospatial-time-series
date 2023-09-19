#!/usr/bin/env python3

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

import argparse

import boto3
import sagemaker
import sagemaker.pytorch
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

if __name__ == "__main__":

    def str2bool(v):
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    # yapf: disable
    parser = argparse.ArgumentParser(description="Pretrain a model using a bunch unlabeled Sentinel-2 time series")
    # SageMaker-related
    parser.add_argument("--branch", type=str, required=False, default="master", help="The GitHub branch to use")
    parser.add_argument("--checkpoint-s3", type=str, required=True, help="The location on S3 where checkpoitns will be deposited")
    # parser.add_argument("--docker-image", type=str, required=True, help="The Docker image to use")
    parser.add_argument("--execution-role", type=str, required=True, help="The SageMaker execution role")
    parser.add_argument("--input-s3", type=str, required=True, help="The location on S3 where the training data are")
    parser.add_argument("--instance-count", type=int, required=True, help="The number of instances to use")
    parser.add_argument("--instance-type", type=str, required=False, default="ml.m5.large", help="The instance type")
    parser.add_argument("--minutes-max", type=int, required=False, default=300, help="The maximum number of minutes to allow the job")
    parser.add_argument("--output-s3", type=str, required=True, help="The location on S3 where outputs will be deposited")
    # Hyperparameters
    parser.add_argument("--architecture", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "mobilenetv3", "efficientnetb0"], help="The model architecture to use (default: resnet18)")
    parser.add_argument("--autocast", type=str, default="float16", choices=["bfloat16", "float16", "float32"], help="The autocast type (default: bfloat16)")
    parser.add_argument("--bands", type=int, nargs="+", default=list(range(0, 12)), help="The Sentinel-2 bands to use (0 indexed)")
    parser.add_argument("--batch-size", type=int, default=7, help="The batch size (default: 7)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="The device to use for training (default: cuda)")
    parser.add_argument("--embeddings-npz", type=str, required=False, default=None, help="Where to find the embeddings")
    parser.add_argument("--epochs", type=int, default=8, help="The number of epochs (default: 8)")
    parser.add_argument("--latent-dims", type=int, default=8, help="The number of shared latent dimensions (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (default: 1e-4)")
    parser.add_argument("--num-workers", type=int, default=3, help="Number of worker processes for the DataLoader (default: 3)")
    parser.add_argument("--pretrained", type=str2bool, default=False, help="Whether to start from pretrained weights (default: False)")
    parser.add_argument("--pth-in", type=str, help="Optional path to a .pth file to use as a starting point for model training")
    parser.add_argument("--pth-out", type=str, default="model.pth", help="The name of the output .pth file (default: model.pth)")
    parser.add_argument("--series-length", type=int, default=8, help="The number of time steps in each sample (default: 8)")
    args = parser.parse_args()
    # yapf: enable

    git_config = {
        "repo": "https://github.com/jamesmcclain/geospatial-time-series.git",
        "branch": args.branch,
    }

    hyperparameters = {
        "checkpoint-dir": "/opt/ml/checkpoints",
        "architecture": args.architecture,
        "autocast": args.autocast,
        "bands": " ".join(map(str, args.bands)),
        "batch-size": args.batch_size,
        "device": args.device,
        "epochs": args.epochs,
        "latent-dims": args.latent_dims,
        "lr": args.lr,
        "num-workers": args.num_workers,
        "pretrained": args.pretrained,
        "pth-out": args.pth_out,
        "series-length": args.series_length,
    }
    if args.embeddings_npz:
        hyperparameters.update(
            {
                "embeddings-npz": args.embeddings_npz,
            }
        )
    if args.pth_in:
        hyperparameters.update(
            {
                "pth-in": args.pth_in,
            }
        )

    pytorch_estimator = sagemaker.pytorch.PyTorch(
        # image_uri=args.docker_image,
        checkpoint_local_path="/opt/ml/checkpoints",
        checkpoint_s3_uri=args.checkpoint_s3,
        # container_entry_point=["python3"],
        entry_point="pretrain.py",
        framework_version="2.0",
        py_version="py310",
        git_config=git_config,
        hyperparameters=hyperparameters,
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        max_run=args.minutes_max * 60,
        role=args.execution_role,
        source_dir="pretrain",
        use_spot=True,
    )

    training_input = TrainingInput(
        s3_data=args.input_s3,
        input_mode="FastFile",
    )

    pytorch_estimator.fit(
        inputs={"train": training_input},
        wait=True,
    )
