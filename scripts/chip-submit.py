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
import logging

import boto3
from sagemaker.processing import ProcessingOutput, ScriptProcessor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep

if __name__ == "__main__":
    logging.basicConfig()
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    # yapf: disable
    parser = argparse.ArgumentParser(description="")
    # SageMaker-related
    parser.add_argument("--docker-image", type=str, required=True, help="The Docker image to use")
    parser.add_argument("--execution-role", type=str, required=True, help="The SageMaker execution role")
    parser.add_argument("--instance-count", type=int, required=True, help="The number of instances to use")
    parser.add_argument("--instance-type", type=str, required=False, default="ml.t3.medium", help="The instance type")
    parser.add_argument("--output", type=str, required=True, help="The location on S3 where outputs will be deposited")
    parser.add_argument("--minutes-max", type=int, required=False, default=20, help="The maximum number of minutes to allow the job")
    # chip-related
    parser.add_argument("--series", type=int, required=True, help="The number of chips to collect")
    parser.add_argument("--good-data-threshold", type=float, required=False, default=0.80, help="The minimum proportion of pixels needed for a chip to be \"good\"")
    parser.add_argument("--size", type=int, required=False, default=512, help="The linear size (in pixels) of each chip")
    parser.add_argument("--bucket", type=str, required=False, default="sentinel-cogs", help="The S3 bucket with the source data")
    parser.add_argument("--prefix", type=str, required=True, help="The S3 prefix for the source data")
    args = parser.parse_args()
    # yapf: enable

    chip_dir = "/opt/ml/processing/chip"
    chip_s3 = f"{args.output}/chip"
    extent_dir = "/opt/ml/processing/extents"
    extent_s3 = f"{args.output}/extents"

    script_arguments = [
        "--series", f"{args.series}",
        "--good-data-threshold", f"{args.good_data_threshold}",
        "--size", f"{args.size}",
        "--output-chip-dir", chip_dir,
        "--output-extent-dir", extent_dir,
        "--bucket", f"{args.bucket}",
        "--prefix", f"{args.prefix}",
    ]

    sagemaker_session = PipelineSession()
    steps = []
    step_processor = ScriptProcessor(
        role=args.execution_role,
        image_uri=args.docker_image,
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        sagemaker_session=sagemaker_session,
        max_runtime_in_seconds=args.minutes_max * 60,
        command=["python3"],
    )
    step_args = step_processor.run(
        inputs=[],
        outputs=[
            ProcessingOutput(source=chip_dir, destination=chip_s3),
            ProcessingOutput(source=extent_dir, destination=extent_s3),
        ],
        code="./chips.py",
        arguments=script_arguments,
    )
    step = ProcessingStep(f"chip", step_args=step_args)
    steps.append(step)

    iam_client = boto3.client("iam")
    role_arn = iam_client.get_role(RoleName=args.execution_role)["Role"]["Arn"]
    pipeline = Pipeline(
        name="timeseries-chip",
        steps=steps,
        sagemaker_session=sagemaker_session,
    )
    pipeline.upsert(role_arn=role_arn)
    execution = pipeline.start()
    print(execution.describe())
