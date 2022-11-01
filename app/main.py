import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import glob
import logging as log

from lib.trainer import Trainer
from lib.options import parse_options

# Set logger display format
log.basicConfig(format='[%(asctime)s] [INFO] %(message)s', 
                datefmt='%d/%m %H:%M:%S',
                level=log.INFO)


if __name__ == "__main__":
    """Main program."""

    args, args_str = parse_options()
    log.info(f'Parameters: \n{args_str}')
    log.info(f'Training on {args.dataset_path}')
    model = Trainer(args, args_str)
    model.train()
