#!/usr/bin/env python3

import argparse
import sagemaker
from sagemaker.processing import ScriptProcessor


if __name__ == "__main__":

    # yapf: disable
    # Command line arguments
    parser = argparse.ArgumentParser(description="")
    # yapf: enable

    script_processor = ScriptProcessor(
        role="AmazonSageMakerExecutionRole",
        image_uri="279682201306.dkr.ecr.us-east-1.amazonaws.com/raster-vision-private:pytorch-f62c367",
        instance_count=1,
        instance_type="ml.m4.xlarge",
        command=["python3"],
    )

    script_processor.run(
        code="hello.py",
        inputs=[],
        outputs=[],
    )

    description = script_processor.jobs[-1].describe()
    print(description)
