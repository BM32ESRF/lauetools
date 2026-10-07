"""Download the LaueTools example images (files attached to a GitHub release, too large for the package).

Same as the command:  lauetools-copy -d FOLDER --download
usage: python download_examples.py [FOLDER]   (default: current directory, images in FOLDER/LaueImages)
"""
import sys

from LaueTools.cli import download_examples

if __name__ == "__main__":
    download_examples(sys.argv[1] if len(sys.argv) > 1 else ".")
