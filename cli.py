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

    with resources.path("Lauetools.scripts", "") as scripts_dir:
        for item in os.listdir(scripts_dir):
            s = os.path.join(scripts_dir, item)
            d = os.path.join(dest, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

    print(f"Great! LaueTools resources copied to: {dest}")

def copy_materials():
    parser = argparse.ArgumentParser(
        description="Copy LaueTools materials.yaml to a user-writable folder."
    )
    parser.add_argument(
        "-d", "--destination",
        default=os.path.join(os.path.expanduser("~"), ".Lauetools"),
        help="Destination folder (default: ~/.Lauetools)"
    )
    args = parser.parse_args()

    dest = os.path.abspath(args.destination)
    os.makedirs(dest, exist_ok=True)

    with resources.path("Lauetools", "materials.yaml") as src:
        dst_file = os.path.join(dest, "materials.yaml")
        shutil.copy2(src, dst_file)

    print(f"Great! materials.yaml copied to: {dst_file}")
    print("You can now edit this file freely.")
