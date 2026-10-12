import argparse
import yaml
import os
from LaueTools.dict_LaueTools import Materials

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
        raise ValueError("Lattice must have 6 values in a string: 'a b c alpha beta gamma', without arithmetical expressions")

    data[label] = {
        "label": label,
        "lattice": [float(x) for x in lattice_parts],
        "extinction": extinction,
    }

    with open(material_file, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"Great! Material '{label}' added/updated in {material_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Add or update a material entry in materials.yaml.\nlauetools-add-material -mat 'Aluminum' -l '4.05 4.05 4.05 90 90 90' -e 'fcc'\nFor the list of possible extinction rules, ask LaueTools's developers and see CrystalParameters.ApplyExtinctionrules() implementation."
    )
    parser.add_argument("-mat", "--material", required=True, help="Material label")
    parser.add_argument(
        "-l", "--lattice",
        required=True,
        help="Lattice parameters: 'a b c alpha beta gamma'"
    )
    parser.add_argument(
        "-e", "--extinction",
        required=True,
        help="Extinction symbol (e.g. 'bcc')"
    )
    parser.add_argument(
        "-file", "--file",
        help="Path to materials.yaml (optional, else uses active one)",
    )

    args = parser.parse_args()

    if args.file:
        add_or_update_material(args.file, args.material, args.lattice, args.extinction)
        return

    # active materials.yaml (written in the user's folder if the active one cannot be modified)
    lattice_parts = args.lattice.split()
    if len(lattice_parts) != 6:
        parser.error("Lattice must have 6 values in a string: 'a b c alpha beta gamma'")
    library = Materials()
    yaml_file = library.add_or_update_material(args.material, [float(x) for x in lattice_parts],
                                               args.extinction)
    print(f"Great! Material '{args.material}' added/updated in {yaml_file}")
