#
# Copyright 2013-2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the
# License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, express or implied. See the License for the specific language governing permissions and
# limitations under the License.
#
# Submits an image classification training job to an AWS Batch job queue, and tails the CloudWatch log output.
#
# https://raw.githubusercontent.com/awslabs/aws-batch-helpers/master/gpu-example/submit-job.py

import argparse
import sys
import time
from datetime import datetime

import boto3
from botocore.compat import total_seconds

batch = boto3.client(
    service_name='batch',
    region_name='us-east-1',
    endpoint_url='https://batch.us-east-1.amazonaws.com')

cloudwatch = boto3.client(
    service_name='logs',
    region_name='us-east-1',
    endpoint_url='https://logs.us-east-1.amazonaws.com')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--name", help="name of the job", type=str, required=True)
parser.add_argument("--job-queue", help="name of the job queue to submit this job", type=str, required=True)
parser.add_argument("--job-definition", help="name of the job job definition", type=str, required=True)
parser.add_argument("--command", help="command to run", type=str, required=True)
# parser.add_argument("--wait", help="block wait until the job completes", action='store_true')

args = parser.parse_args()


def main():
    jobName = args.name
    jobQueue = args.job_queue
    jobDefinition = args.job_definition
    command = args.command.split()

    submitJobResponse = batch.submit_job(
        jobName=jobName,
        jobQueue=jobQueue,
        jobDefinition=jobDefinition,
        containerOverrides={'command': command}
    )

    jobId = submitJobResponse['jobId']
    print(f'Submitted job [{jobName} - {jobId}] to the job queue [{jobQueue}]')


if __name__ == "__main__":
    main()
