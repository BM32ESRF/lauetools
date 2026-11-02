# -*- coding: utf-8 -*-
"""
Read crystal structures from CIF files (Crystallographic Information File) to create LaueTools materials.

    import LaueTools.cifreader as CIF
    label, lattice, extinction = CIF.material_from_cif("Si.cif")   # e.g. 'Si', [5.43088, ...], '227:1'
    import LaueTools.dict_LaueTools as DictLT
    DictLT.dict_Materials.add_or_update_material(label, lattice, extinction)

The extinction code is the space group number with its setting when the CIF gives one ('227:1', '167:H', '14:b'):
LaueTools then applies the general reflection conditions of this space group (wyckpos_lauetools.py).

Self-contained parser (no dependency) of the CIF 1.1 syntax: data blocks, tags and values, loops, quoted values,
multi-line text fields; numbers with standard uncertainty (5.43088(5)).
"""
import os
import re

from . import wyckpos_lauetools as WYCK

CELL_TAGS = ("_cell_length_a", "_cell_length_b", "_cell_length_c",
             "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma")
SPACEGROUP_NUMBER_TAGS = ("_space_group_it_number", "_symmetry_int_tables_number")
SPACEGROUP_SYMBOL_TAGS = ("_space_group_name_h-m_alt", "_symmetry_space_group_name_h-m")
SYMOP_TAGS = ("_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz")
NAME_TAGS = ("_chemical_formula_structural", "_chemical_formula_sum", "_chemical_name_mineral",
             "_chemical_name_systematic", "_chemical_name_common")


# --- -------------------- CIF syntax
def _tokens(text):
    """values and keywords of a CIF text, in order (comments removed)"""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(";"):  # multi-line text field, ended by a line starting with ';'
            field = [line[1:]]
            i += 1
            while i < len(lines) and not lines[i].startswith(";"):
                field.append(lines[i])
                i += 1
            yield "\n".join(field).strip()
            i += 1
            continue
        pos = 0
        while pos < len(line):
            char = line[pos]
            if char.isspace():
                pos += 1
            elif char == "#":
                break
            elif char in "'\"":  # quoted value: ends with the same quote followed by a blank or the line end
                end = pos + 1
                while end < len(line) and not (line[end] == char and (end + 1 == len(line)
                                                                      or line[end + 1].isspace())):
                    end += 1
                yield line[pos + 1:end]
                pos = end + 1
            else:
                end = pos
                while end < len(line) and not line[end].isspace():
                    end += 1
                yield line[pos:end]
                pos = end
        i += 1


def parse_cif(text):
    """data blocks of a CIF text

    :return: list of dict {'name': block name, 'items': {tag: value}, 'loops': [{tag: [values]}]}, tags in lower
        case with '.' replaced by '_' (CIF 2 / mmCIF style _cell.length_a -> _cell_length_a)
    """
    blocks = []
    tokens = list(_tokens(text))
    i = 0
    block = None
    norm = lambda tag: tag.lower().replace(".", "_")
    while i < len(tokens):
        token = tokens[i]
        lower = token.lower()
        if lower.startswith("data_"):
            block = {"name": token[5:], "items": {}, "loops": []}
            blocks.append(block)
            i += 1
        elif block is None:
            i += 1
        elif lower == "loop_":
            i += 1
            tags = []
            while i < len(tokens) and tokens[i].startswith("_"):
                tags.append(norm(tokens[i]))
                i += 1
            values = []
            while i < len(tokens) and not tokens[i].startswith("_") and tokens[i].lower() != "loop_" \
                    and not tokens[i].lower().startswith("data_"):
                values.append(tokens[i])
                i += 1
            if tags:
                block["loops"].append({tag: values[k::len(tags)] for k, tag in enumerate(tags)})
        elif token.startswith("_"):
            if i + 1 < len(tokens):
                block["items"][norm(token)] = tokens[i + 1]
            i += 2
        else:
            i += 1
    return blocks


def cif_number(value):
    """float of a CIF number ('5.43088(5)' -> 5.43088), None for '?' or '.' (unknown, not applicable)"""
    if value is None or value in ("?", "."):
        return None
    return float(re.sub(r"\(\d+\)$", "", value.strip()))


def _get(block, tags):
    """value of the first tag of tags found in the block items or (first row) in its loops"""
    for tag in tags:
        if tag in block["items"]:
            return block["items"][tag]
        for loop in block["loops"]:
            if tag in loop and loop[tag]:
                return loop[tag][0]
    return None


def _loop(block, tag):
    """loop of the block containing tag, or None"""
    for loop in block["loops"]:
        if tag in loop:
            return loop
    return None


# --- -------------------- crystal structure
def spacegroup_setting(number, symbol):
    """setting suffix of wyckpos_lauetools ('', ':1', ':2', ':H', ':R', ':b', ':c') from the Hermann-Mauguin
    symbol of the CIF ('F d -3 m :1', 'R -3 c :H', 'P 1 21/c 1'), '' (default setting) if not found"""
    possible = WYCK.get_possible_sgrp_suf(number) or []
    if not possible or not symbol:
        return ""
    match = re.search(r":\s*([12HRhr])\s*$", symbol)
    if match and ":" + match.group(1).upper() in possible:
        return ":" + match.group(1).upper()
    if possible == [":b", ":c"]:  # monoclinic: unique axis from the full symbol 'P 1 21/c 1' or 'P 1 1 21/b'
        parts = symbol.split()
        if len(parts) == 4:
            return ":b" if parts[1] == "1" and parts[3] == "1" else ":c"
    return ""


def element_of(type_symbol):
    """chemical element of an atom type symbol of CIF ('O2-' -> 'O', 'Fe3+' -> 'Fe', 'Si1' -> 'Si')"""
    match = re.match(r"([A-Z][a-z]?)", type_symbol.strip())
    return match.group(1) if match else type_symbol


def read_cif(path, block_name=None):
    """crystal structure of a CIF file

    :param path: path of the .cif file
    :param block_name: name of the data block (data_<name>), default: the first one with a unit cell
    :return: dict with
        name: str, chemical name or formula (or data block name)
        lattice: [a, b, c, alpha, beta, gamma] (Angstrom, degrees)
        spacegroup: int or None, spacegroup_symbol: str or None, setting: str ('', ':1', ':H', ...)
        symops: list of symmetry operations ('x, y, z', '-x+1/2, ...'), possibly empty
        atoms: list of dict(label, element, x, y, z, occupancy) of the asymmetric unit
    :raises ValueError: if no data block has a unit cell
    """
    with open(path, encoding="utf-8", errors="replace") as cif:
        blocks = parse_cif(cif.read())
    candidates = [b for b in blocks if (block_name is None or b["name"] == block_name)
                  and _get(b, CELL_TAGS[:1]) is not None]
    if not candidates:
        raise ValueError("%s: no data block with a unit cell (_cell_length_a)" % path)
    block = candidates[0]

    lattice = [cif_number(_get(block, (tag,))) for tag in CELL_TAGS]
    if any(value is None for value in lattice):
        raise ValueError("%s: incomplete unit cell %s" % (path, lattice))

    number = _get(block, SPACEGROUP_NUMBER_TAGS)
    number = int(cif_number(number)) if cif_number(number) is not None else None
    symbol = _get(block, SPACEGROUP_SYMBOL_TAGS)
    setting = spacegroup_setting(number, symbol) if number else ""
    if not setting and number and WYCK.get_possible_sgrp_suf(number) == [":H", ":R"]:
        # rhombohedral space group: hexagonal axes (gamma = 120) or rhombohedral axes (a = b = c)
        setting = ":H" if abs(lattice[5] - 120) < 1e-3 else ":R"

    symops = []
    for tag in SYMOP_TAGS:
        loop = _loop(block, tag)
        if loop:
            symops = list(loop[tag])
            break

    atoms = []
    loop = _loop(block, "_atom_site_fract_x")
    if loop:
        nb = len(loop["_atom_site_fract_x"])
        labels = loop.get("_atom_site_label", [""] * nb)
        types = loop.get("_atom_site_type_symbol", labels)
        occupancies = loop.get("_atom_site_occupancy", ["1"] * nb)
        for k in range(nb):
            occupancy = cif_number(occupancies[k])
            atoms.append({"label": labels[k], "element": element_of(types[k]),
                          "x": cif_number(loop["_atom_site_fract_x"][k]),
                          "y": cif_number(loop["_atom_site_fract_y"][k]),
                          "z": cif_number(loop["_atom_site_fract_z"][k]),
                          "occupancy": 1.0 if occupancy is None else occupancy})

    name = _get(block, NAME_TAGS) or block["name"]
    return {"name": name, "lattice": lattice, "spacegroup": number, "spacegroup_symbol": symbol,
            "setting": setting, "symops": symops, "atoms": atoms}


def material_label(text):
    """material label (no blank) from a chemical name or formula: 'Al2 O3' -> 'Al2O3'"""
    return re.sub(r"\s+", "", text.strip()) or "material"


def material_from_cif(path, label=None, block_name=None):
    """[label, lattice, extinction] of a LaueTools material from a CIF file

    :param label: material label, default: chemical name or formula of the CIF (blanks removed)
    :return: label, [a, b, c, alpha, beta, gamma], extinction code: space group number with its setting
        ('227:1', '167:H'), or 'no' if the CIF gives no space group number
    """
    structure = read_cif(path, block_name)
    if label is None:
        label = material_label(structure["name"]) if structure["name"] else \
            os.path.splitext(os.path.basename(path))[0]
    if structure["spacegroup"]:
        extinction = "%d%s" % (structure["spacegroup"], structure["setting"])
    else:
        extinction = "no"
    return label, structure["lattice"], extinction
