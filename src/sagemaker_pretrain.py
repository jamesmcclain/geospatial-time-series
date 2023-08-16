#!/usr/bin/env python3

import argparse
import sagemaker
import sagemaker.pytorch

# sagemaker_session = sagemaker.Session()
# role = sagemaker.get_execution_role()

if __name__ == "__main__":

    def str2bool(v):
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    # yapf: disable
    # Command line arguments
    parser = argparse.ArgumentParser(description="Pretrain a model using a bunch unlabeled Sentinel-2 time series")
    parser.add_argument("cog_dirs", nargs="+", type=str, help="Paths to the data")
    parser.add_argument("--architecture", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "mobilenetv3", "efficientnetb0"], help="The model architecture to use (default: resnet18)")
    parser.add_argument("--bands", type=int, nargs="+", default=list(range(1, 12 + 1)), help="The Sentinel-2 bands to use (1 indexed)")
    parser.add_argument("--batch-size", type=int, default=7, help="The batch size (default: 7)")
    parser.add_argument("--dataset", type=str, required=True, choices=["embed-series", "series", "digest"], help="The type of data found in the data directories")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="The device to use for training (default: cuda)")
    parser.add_argument("--epochs", type=int, default=8, help="The number of epochs (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (default: 1e-4)")
    parser.add_argument("--pretrained", type=str2bool, default=False, help="Whether to start from pretrained weights (default: False)")
    parser.add_argument("--num-workers", type=int, default=3, help="Number of worker processes for the DataLoader (default: 3)")
    parser.add_argument("--output-dir", type=str, required=True, help="The directory where logs and artifacts will be deposited")
    parser.add_argument("--pth-in", type=str, help="Optional path to a .pth file to use as a starting point for model training")
    parser.add_argument("--pth-out", type=str, default="model.pth", help="The name of the output .pth file (default: model.pth)")
    parser.add_argument("--series-length", type=int, default=8, help="The number of time steps in each sample (default: 8)")
    parser.add_argument("--latent-dims", type=int, default=8, help="The number of shared latent dimensions (default: 8)")
    parser.add_argument("--size", type=int, default=512, help="The tile size (default: 512)")
    # yapf: enable

    args = parser.parse_args()

    git_config = {
        "repo": "https://github.com/jamesmcclain/geospatial-time-series.git",
        "branch": "organize",
    }

    hyperparameters = {
        "cog_dirs": ["this", "space", "intentionally", "left", "blank"],
        "architecture": args.architecture,
        "bands": args.bands,
        "batch-size": args.batch_size,
        "dataset": args.dataset,
        "device": args.device,
        "epochs": args.epochs,
        "lr": args.lr,
        "pretrained": args.pretrained,
        "num-workers": args.num_workers,
        "pth-in": args.pth_in,
        "pth-out": args.pth_out,
        "series-length": args.series_length,
        "latent-dims": args.latent_dims,
        "size": args.size,
    }

    pytorch_estimator = sagemaker.pytorch.PyTorch(
        entry_point="pretrain.py",
        framework_version="2.0",
        git_config=git_config,
        hyperparameters=hyperparameters,
        instance_count=1,
        instance_type="ml.p3.2xlarge",
        output_path=args.output_dir,
        py_version="py310",
        role="AmazonSageMakerExecutionRole",
        source_dir="src/unsupervised_pretrain",
        use_spot=True,
    )

    pytorch_estimator.fit(
        {"train": args.cog_dirs[0]},
        wait=True,
    )
