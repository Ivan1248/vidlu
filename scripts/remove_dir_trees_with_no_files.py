import argparse
from pathlib import Path

from _context import vidlu

from vidlu.utils.path import remove_dir_trees_with_no_files

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
args = parser.parse_args()

remove_dir_trees_with_no_files(Path(args.path), verbose=True)
