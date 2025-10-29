import argparse
import yaml
import os
from LaueTools.dict_LaueTools import get_materials_file

def add_or_update_material(material_file, label, lattice, extinction):
    """
    Add or update a material in materials.yaml.
    Lattice should be a string 'a b c alpha beta gamma'.
    """
    # Normalize the path
    material_file = os.path.abspath(material_file)
    os.makedirs(os.path.dirname(material_file), exist_ok=True)

    # Load existing data if present
    if os.path.exists(material_file):
        with open(material_file, "r") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    # Parse lattice string
    lattice_parts = lattice.split()
    if len(lattice_parts) != 6:
        raise ValueError("Lattice must have 6 values: a b c alpha beta gamma")

    data[label] = {
        "lattice": [float(x) for x in lattice_parts],
        "extinction": extinction,
    }

    with open(material_file, "w") as f:
        yaml.safe_dump(data, f, sort_keys=True)

    print(f"Great! Material '{label}' added/updated in {material_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Add or update a material entry in materials.yaml."
    )
    parser.add_argument("-mat", "--material", required=True, help="Material label")
    parser.add_argument(
        "-l", "--lattice",
        required=True,
        help="Lattice parameters: a b c alpha beta gamma"
    )
    parser.add_argument(
        "-e", "--extinction",
        required=True,
        help="Extinction symbol (e.g. Fm-3m)"
    )
    parser.add_argument(
        "-file", "--file",
        help="Path to materials.yaml (optional, else uses active one)",
    )

    args = parser.parse_args()

    # Determine which materials.yaml to use
    yaml_file = args.file or get_materials_file()
    add_or_update_material(yaml_file, args.material, args.lattice, args.extinction)
