#!/usr/bin/env python3

import argparse
import os
import json


if __name__ == "__main__":

    # yapf: disable
    # Command line arguments
    parser = argparse.ArgumentParser(description="")
    # yapf: enable

    print(json.dumps(dict(os.environ), indent=4))
