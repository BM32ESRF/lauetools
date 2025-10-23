import os
import shutil
import argparse
from importlib import resources

def copy_resources():
    parser = argparse.ArgumentParser(
        description="Copy LaueTools example notebooks and scripts to a folder."
    )
    parser.add_argument(
        "-d", "--destination", default=os.getcwd(),
        help="Destination folder (default: current directory)"
    )
    args = parser.parse_args()

    dest = os.path.abspath(args.destination)
    os.makedirs(dest, exist_ok=True)

    with resources.path("lauetools.data", "") as data_dir, \
         resources.path("lauetools.scripts", "") as scripts_dir:

        for src_dir in [data_dir, scripts_dir]:
            for item in os.listdir(src_dir):
                s = os.path.join(src_dir, item)
                d = os.path.join(dest, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)

    print(f"Great! LaueTools resources copied to: {dest}")
