"""
module of lauetools project

Segmentation of a 2D Laue map into clusters of similar orientations (grains)
from the UB matrices stored in .fit files, e.g. eiger4m_0123_g0.fit, eiger4m_0123_g1.fit

- UB matrices are orthonormalised (polar decomposition) before any angle computation
- misorientation angles take into account the crystal symmetry (24 proper rotations for cubic)
- two matrices are linked if they belong to neighbouring (or the same) map points and
  their misorientation is below a threshold; clusters are the connected components
- per-cluster statistics, cluster maps, single-cluster maps, KAM and GROD maps
  (whole map or a single cluster on demand)
- grain envelopes of sparse clusters (morphological closing / convex hull), superimposed main
  clusters, boundary map colored by misorientation, table of neighbouring cluster pairs
- check_symmetry_equivalents() warns when matrices look different but are equivalent by symmetry

Map convention (same as reshape(invmapdims) elsewhere in LaueTools notebooks):
    mapdimension = (nrows, ncols) = (nb points along slow motor, nb points along fast motor)
    image index = row * ncols + col

Typical use in a notebook:

    import LaueTools.orientationclustering as OC
    result, analyzer = OC.analyze_ub_matrices(fitfilefolder, threshold=1.0,
                                              mapdimension=invmapdims, prefix='eiger4m_',
                                              symmetry='cubic', min_cluster_size=8)
    OC.plot_cluster_map(result, analyzer)
    OC.plot_clusters_gallery(result, analyzer, min_size=30)
    kam = OC.compute_kam(analyzer, result)
    OC.plot_kam(kam, result, analyzer)
    OC.plot_cluster_grod_kam(result, analyzer, cluster_id=4)       # GROD + KAM of one cluster
    OC.plot_cluster_envelopes(result, analyzer, max_clusters=10)    # main clusters superimposed
    OC.plot_boundary_map(result, analyzer)
    OC.cluster_pairs_table(result, analyzer)

JS Micha Sept 2026
"""
import re
import string
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import dict_LaueTools as DictLT
from . import generaltools as GT


# =============================================================================
# SYMMETRY AND MISORIENTATION
# =============================================================================

_OPSYM = np.array(DictLT.OpSymArray, dtype=float)
#: 24 proper rotations of the cubic point group (det = +1)
CUBIC_ROTATIONS = _OPSYM[np.isclose(np.linalg.det(_OPSYM), 1.0)]
IDENTITY_OPS = np.eye(3)[np.newaxis]


def get_symmetry_operators(symmetry: Union[None, str, np.ndarray] = 'cubic') -> np.ndarray:
    """return array (n, 3, 3) of proper rotations S such that UB and UB.S are equivalent

    symmetry: 'cubic' (24 operators), None or 'none' (identity only), or an array of operators
    """
    if symmetry is None or (isinstance(symmetry, str) and symmetry.lower() == 'none'):
        return IDENTITY_OPS
    if isinstance(symmetry, str):
        if symmetry.lower() == 'cubic':
            return CUBIC_ROTATIONS
        raise ValueError(f"Unknown symmetry '{symmetry}'. Use 'cubic', None or an array of operators")
    ops = np.asarray(symmetry, dtype=float).reshape(-1, 3, 3)
    return ops


def orthonormalize(ub: np.ndarray) -> np.ndarray:
    """return the rotation closest to ub (polar decomposition), works on (..., 3, 3) arrays

    NB: np.linalg.qr is not suited: it biases the result towards the first column and can
    return sign-flipped matrices
    matrices containing NaN (e.g. missing .fit file) give NaN matrices
    """
    ub = np.asarray(ub, dtype=float)
    finite = np.all(np.isfinite(ub), axis=(-2, -1))
    if not np.all(finite):
        rot = np.full(ub.shape, np.nan)
        rot[finite] = orthonormalize(ub[finite])
        return rot
    u, _, vt = np.linalg.svd(ub)
    rot = u @ vt
    det = np.linalg.det(rot)
    if np.any(det < 0):  # improper matrix: flip the last singular direction
        u = np.where((det < 0)[..., None, None], u * np.array([1, 1, -1]), u)
        rot = u @ vt
    return rot


def misorientation(rotA: np.ndarray, rotB: np.ndarray,
                   symops: np.ndarray = CUBIC_ROTATIONS) -> Tuple[np.ndarray, np.ndarray]:
    """misorientation angle (deg) between rotations rotA and rotB, minimised over symops

    rotA, rotB: (3, 3) or (N, 3, 3) arrays of ORTHONORMAL matrices (see orthonormalize())
    symops: (n, 3, 3) symmetry operators S (rotB and rotB.S are equivalent)

    return: angles (N,) in deg, index of best operator (N,) such that rotB.S[k] is the
            symmetry variant of rotB closest to rotA
    """
    rotA = np.asarray(rotA, dtype=float).reshape(-1, 3, 3)
    rotB = np.asarray(rotB, dtype=float).reshape(-1, 3, 3)
    # R = A^T B ;  trace(R S) = sum_ij R_ij S_ji
    rel = np.einsum('nki,nkj->nij', rotA, rotB)
    traces = np.einsum('nij,sji->ns', rel, symops)
    best = np.argmax(traces, axis=1)
    cosang = np.clip((traces[np.arange(len(best)), best] - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosang)), best


def misorientation_axis(rotA: np.ndarray, rotB: np.ndarray,
                        symops: np.ndarray = CUBIC_ROTATIONS) -> Tuple[np.ndarray, np.ndarray]:
    """misorientation angle (deg) and unit rotation axis (crystal frame of A) of the
    symmetry variant of minimum angle. rotA, rotB: (3, 3) or (N, 3, 3) orthonormal matrices"""
    rotA = np.asarray(rotA, dtype=float).reshape(-1, 3, 3)
    rotB = np.asarray(rotB, dtype=float).reshape(-1, 3, 3)
    angles, best = misorientation(rotA, rotB, symops)
    rel = np.einsum('nki,nkj->nij', rotA, rotB) @ symops[best]
    axis = np.column_stack([rel[:, 2, 1] - rel[:, 1, 2], rel[:, 0, 2] - rel[:, 2, 0], rel[:, 1, 0] - rel[:, 0, 1]])
    norm = np.linalg.norm(axis, axis=1, keepdims=True)
    return angles, np.divide(axis, norm, out=np.zeros_like(axis), where=norm > 1e-12)


def _sigma3_rotations() -> np.ndarray:
    """the 8 rotations of +-60 deg about the 4 <111> axes (cubic Sigma3 twin relation)"""
    mats = []
    for ax in ([1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]):
        for ang in (60.0, -60.0):
            mats.append(GT.matRot(np.array(ax, dtype=float), ang))
    return np.array(mats)


SIGMA3_ROTATIONS = _sigma3_rotations()
#: Brandon criterion for Sigma3: max deviation 15 / sqrt(3) deg
BRANDON_SIGMA3 = 15.0 / np.sqrt(3.0)


def sigma3_deviation(rotA: np.ndarray, rotB: np.ndarray) -> np.ndarray:
    """deviation (deg) of the misorientation between rotA and rotB from the ideal cubic Sigma3
    twin relation (60 deg about <111>). Twin if deviation < BRANDON_SIGMA3 (8.66 deg)"""
    rotA = np.asarray(rotA, dtype=float).reshape(-1, 3, 3)
    rotB = np.asarray(rotB, dtype=float).reshape(-1, 3, 3)
    rel = np.einsum('nki,nkj->nij', rotA, rotB)
    devs = [misorientation(np.broadcast_to(T, rel.shape), rel, CUBIC_ROTATIONS)[0] for T in SIGMA3_ROTATIONS]
    return np.min(devs, axis=0)


def misorientation_angle(UB_A: np.ndarray, UB_B: np.ndarray, symmetry='cubic') -> float:
    """convenience: misorientation angle (deg) between two (non-orthonormal) UB matrices"""
    angle, _ = misorientation(orthonormalize(UB_A), orthonormalize(UB_B), get_symmetry_operators(symmetry))
    return float(angle[0])


def rotation_angle_between_matrices(R1: np.ndarray, R2: np.ndarray, symmetry=None) -> float:
    """rotation angle (deg) between two matrices (orthonormalised first). symmetry=None: raw angle"""
    return misorientation_angle(R1, R2, symmetry=symmetry)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class UBMatrix:
    """a single UB matrix with metadata

    matrix: UB as written in .fit file ; rotation: its closest rotation (orthonormalised)
    """
    matrix: np.ndarray
    image_index: int
    grain_index: int
    filename: str
    rotation: Optional[np.ndarray] = None
    material: str = ''
    source_grain: Optional[int] = None
    pixdev: Optional[float] = None
    nb_indexed: int = 0
    # quantities of the grain block in its .fit file: 'B0', 'strain_crystal', 'strain_sample'
    # (absolute strain), 'latticeparams', 'euler' (as written, see matrix_properties())
    props: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.rotation is None:
            self.rotation = orthonormalize(self.matrix)

    def __repr__(self):
        return (f"UBMatrix(image={self.image_index}, grain={self.grain_index}, "
                f"material={self.material}, nb_indexed={self.nb_indexed}, file={self.filename})")


@dataclass
class Cluster:
    """a cluster of UB matrices"""
    cluster_id: int
    matrix_indices: List[int]
    image_indices: List[int]
    grain_indices: List[int]
    size: int = field(init=False)

    def __post_init__(self):
        self.size = len(self.matrix_indices)

    @property
    def nb_pixels(self) -> int:
        """nb of distinct map points (a map point may hold several matrices of the cluster)"""
        return len(set(self.image_indices))

    def __repr__(self):
        return f"Cluster({self.cluster_id}, size={self.size}, nb_pixels={self.nb_pixels})"


@dataclass
class ClusterStats:
    """statistics of a cluster, orientations referred to the matrix closest to the barycenter

    positions are (row, col) = (slow axis, fast axis)
    angles are symmetry-reduced misorientations (deg) from the reference matrix
    """
    cluster_id: int
    size: int
    nb_pixels: int
    reference_matrix: np.ndarray
    reference_matrix_index: int
    reference_image_index: int
    mean_orientation: np.ndarray
    mean_position: Tuple[float, float]
    min_position: Tuple[float, float]
    max_position: Tuple[float, float]
    std_orientation: float
    mean_angular_deviation: float
    max_angular_deviation: float
    image_indices: List[int]
    matrix_indices: List[int]
    grain_indices: List[int]
    materials: List[str]
    all_angles: List[float] = field(default_factory=list)

    def contains_image(self, image_index: int) -> bool:
        return image_index in self.image_indices

    def contains_matrix(self, matrix_index: int) -> bool:
        return matrix_index in self.matrix_indices

    def width(self) -> float:
        """bounding box width (along fast axis, columns)"""
        return self.max_position[1] - self.min_position[1] + 1

    def height(self) -> float:
        """bounding box height (along slow axis, rows)"""
        return self.max_position[0] - self.min_position[0] + 1

    def aspect_ratio(self) -> float:
        h = self.height()
        return self.width() / h if h > 0 else 0.0

    def __repr__(self):
        row, col = self.mean_position
        return (f"ClusterStats(id={self.cluster_id}, size={self.size}, pixels={self.nb_pixels}, "
                f"ref_img={self.reference_image_index}, mean (row, col)=({row:.1f}, {col:.1f}), "
                f"std_orient={self.std_orientation:.2f}°, max_dev={self.max_angular_deviation:.2f}°)")


# =============================================================================
# READING .fit FILES
# =============================================================================

def _read_matrix_block(lines: List[str], i: int) -> np.ndarray:
    """3x3 matrix written on the 3 lines following line i ('#[[ 1. 2. 3.]' ...)"""
    rows = []
    for j in range(1, 4):
        vals = lines[i + j].replace("#", "").replace("[", "").replace("]", "").split()
        rows.append([float(x) for x in vals[:3]])
    return np.array(rows, dtype=float)


def _read_vector_block(lines: List[str], i: int) -> np.ndarray:
    """1D array written after line i, possibly wrapped on several lines, ending with ']'"""
    txt = ''
    for j in range(i + 1, min(i + 4, len(lines))):
        txt += ' ' + lines[j].replace("#", "")
        if ']' in lines[j]:
            break
    return np.array([float(x) for x in txt.replace("[", "").replace("]", "").replace(",", " ").split()])


# header of blocks in .fit files -> (key, reader, factor)
_FIT_BLOCKS = {
    "#B0 matrix": ('B0', _read_matrix_block, 1.),
    "#deviatoric strain in direct crystal frame (10-3 unit)": ('strain_crystal', _read_matrix_block, 1e-3),
    "#deviatoric strain in crystal frame (10-3 unit)": ('strain_crystal', _read_matrix_block, 1e-3),
    "#deviatoric strain in sample2 frame (10-3 unit)": ('strain_sample', _read_matrix_block, 1e-3),
    "#new lattice parameters": ('latticeparams', _read_vector_block, 1.),
    "#Euler angles phi theta psi (deg)": ('euler', _read_vector_block, 1.),
}


def read_fitfile_grains(fitfilename: Union[str, Path]) -> List[Dict[str, Any]]:
    """read all grains of a .fit file

    return list of dict with keys: 'UB' (3x3), 'material', 'pixdev', 'nb_indexed' and 'props':
    dict of the grain block quantities found: 'B0', 'strain_crystal', 'strain_sample' (absolute strain,
    i.e. file values x 1e-3), 'latticeparams' (a, b, c, alpha, beta, gamma), 'euler' (phi, theta, psi)
    """
    with open(fitfilename, 'r') as f:
        lines = [line.rstrip(string.whitespace) for line in f]

    grains = []
    current = None
    for i, line in enumerate(lines):
        if line.startswith(("# Number of indexed spots", "#Number of indexed spots")):
            current = {'UB': None, 'material': '', 'pixdev': None, 'nb_indexed': 0, 'props': {}}
            grains.append(current)
            try:
                current['nb_indexed'] = int(line.split(":")[-1])
            except ValueError:
                pass
        elif current is None:
            continue
        elif line.startswith("#Element") and i + 1 < len(lines):
            current['material'] = lines[i + 1].lstrip('#').strip()
        elif line.startswith(("# Mean Pixel Deviation", "#Mean Deviation", "#Mean Pixel Deviation")):
            try:
                current['pixdev'] = float(line.split(":")[-1])
            except ValueError:
                pass
        elif line.startswith(("#UB matrix", "UB matrix")) and current['UB'] is None:
            current['UB'] = _read_matrix_block(lines, i)
        else:
            for header, (key, reader, factor) in _FIT_BLOCKS.items():
                if line.startswith(header) and key not in current['props']:
                    try:
                        current['props'][key] = reader(lines, i) * factor
                    except (ValueError, IndexError):
                        pass
                    break

    return [g for g in grains if g['UB'] is not None]


@dataclass
class UBMatrixAnalyzer:
    """collection of UB matrices of a 2D map read from a folder of .fit files

    input_dir: folder of .fit files
    prefix: file prefix, e.g. 'eiger4m_' for eiger4m_0123_g0.fit
    mapdimension: (nrows, ncols) i.e. (slow axis, fast axis)
    nbfiles_per_folder: None: .fit files in input_dir. n: .fit files in its subfolders <subfolder_prefix><start>_<end>
        (see indexing_batch.fitfiles_layout()). UBMatrix.filename is then relative to input_dir
    """
    input_dir: Union[str, Path]
    prefix: str = ''
    mapdimension: Optional[Tuple[int, int]] = None
    nbfiles_per_folder: Optional[int] = None
    subfolder_prefix: str = 'images_'
    matrices: List[UBMatrix] = field(default_factory=list)

    def __post_init__(self):
        self.input_dir = Path(self.input_dir)

    def load_matrices(self, verbose: int = 0) -> int:
        """load UB matrices of all .fit files. return nb of matrices"""
        pattern = re.compile(rf"{re.escape(self.prefix)}(\d+)(?:_g(\d+))?\.fit$")
        self.matrices = []
        nb_skipped = 0
        pattern_glob = f'{self.prefix}*.fit'
        if self.nbfiles_per_folder:
            pattern_glob = f'{self.subfolder_prefix}*/{pattern_glob}'
        for filepath in sorted(self.input_dir.glob(pattern_glob), key=lambda fp: fp.name):
            match = pattern.search(filepath.name)
            if match is None:
                nb_skipped += 1
                continue
            image_index = int(match.group(1))
            file_grain = int(match.group(2)) if match.group(2) is not None else None
            try:
                grains = read_fitfile_grains(filepath)
            except (OSError, ValueError, IndexError) as err:
                print(f"Error reading {filepath}: {err}")
                continue
            for k, grain in enumerate(grains):
                self.matrices.append(UBMatrix(matrix=grain['UB'],
                                              image_index=image_index,
                                              # grain index from filename (_g1.fit) else order in file
                                              grain_index=file_grain if file_grain is not None else k,
                                              filename=str(filepath.relative_to(self.input_dir)),
                                              material=grain['material'],
                                              source_grain=k,
                                              pixdev=grain['pixdev'],
                                              nb_indexed=grain['nb_indexed'],
                                              props=grain['props']))
        if verbose and nb_skipped:
            print(f"{nb_skipped} .fit files skipped (name does not match {pattern.pattern})")
        self._build_arrays()
        return len(self.matrices)

    def _build_arrays(self):
        """numpy arrays for fast vectorised computations"""
        self.rotations = np.array([m.rotation for m in self.matrices]).reshape(-1, 3, 3)
        self.image_indices = np.array([m.image_index for m in self.matrices], dtype=int)
        self.grain_indices = np.array([m.grain_index for m in self.matrices], dtype=int)
        self.nb_indexed = np.array([m.nb_indexed for m in self.matrices], dtype=int)
        self.materials = np.array([m.material for m in self.matrices])
        self.image_to_matrices = {}
        for idx, img in enumerate(self.image_indices):
            self.image_to_matrices.setdefault(int(img), []).append(idx)
        self.grain_to_matrices = {}
        for idx, g in enumerate(self.grain_indices):
            self.grain_to_matrices.setdefault(int(g), []).append(idx)

    def check_mapdimension(self, mapdimension: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """return (nrows, ncols), raise error if unknown or inconsistent with image indices"""
        mapdim = mapdimension if mapdimension is not None else self.mapdimension
        if mapdim is None:
            raise ValueError("mapdimension=(nrows, ncols) must be given, i.e. invmapdims = "
                             "(nb points slow axis, nb points fast axis). It is not guessed any more "
                             "(guessing fails when some images have no .fit file)")
        nrows, ncols = int(mapdim[0]), int(mapdim[1])
        if len(self.image_indices) and self.image_indices.max() >= nrows * ncols:
            raise ValueError(f"image index {self.image_indices.max()} out of map {nrows}x{ncols}. "
                             "Check mapdimension or image numbering (first image must be 0)")
        return nrows, ncols

    def positions(self, mapdimension: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """(rows, cols) arrays of all matrices"""
        _, ncols = self.check_mapdimension(mapdimension)
        return self.image_indices // ncols, self.image_indices % ncols

    def get_matrices_for_grain(self, grain_index: int) -> List[UBMatrix]:
        return [self.matrices[i] for i in self.grain_to_matrices.get(grain_index, [])]

    def get_matrices_for_image(self, image_index: int) -> List[UBMatrix]:
        return [self.matrices[i] for i in self.image_to_matrices.get(image_index, [])]

    def get_all_image_indices(self) -> List[int]:
        return sorted(self.image_to_matrices.keys())

    def get_all_grain_indices(self) -> List[int]:
        return sorted(self.grain_to_matrices.keys())


# =============================================================================
# NEIGHBOURHOOD
# =============================================================================

def _forward_offsets(connectivity: Union[str, int] = '8') -> List[Tuple[int, int]]:
    """half of the neighbourhood offsets (drow, dcol), so that each pair is visited once

    connectivity: '4' (edges), '8' (edges + corners) or an int d: all points at Chebyshev distance <= d
    """
    if str(connectivity) == '4':
        return [(0, 1), (1, 0)]
    dist = 1 if str(connectivity) == '8' else int(connectivity)
    return [(dr, dc) for dr in range(0, dist + 1) for dc in range(-dist, dist + 1)
            if (dr > 0 or dc > 0)]


def neighbour_pairs(analyzer: UBMatrixAnalyzer,
                    mapdimension: Optional[Tuple[int, int]] = None,
                    connectivity: Union[str, int] = '8',
                    subset: Optional[np.ndarray] = None,
                    same_image: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """all pairs (i, j) of matrix indices located at the same or at neighbouring map points

    subset: optional array of matrix indices to consider (others are ignored)
    """
    nrows, ncols = analyzer.check_mapdimension(mapdimension)
    if subset is None:
        img_to_mat = analyzer.image_to_matrices
    else:
        img_to_mat = {}
        for idx in subset:
            img_to_mat.setdefault(int(analyzer.image_indices[idx]), []).append(int(idx))

    pairs_i, pairs_j = [], []
    offsets = _forward_offsets(connectivity)
    for img, mats in img_to_mat.items():
        row, col = divmod(img, ncols)
        if same_image:
            for a in range(len(mats)):
                for b in range(a + 1, len(mats)):
                    pairs_i.append(mats[a])
                    pairs_j.append(mats[b])
        for dr, dc in offsets:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < nrows and 0 <= nc < ncols):
                continue
            neigh = img_to_mat.get(nr * ncols + nc)
            if neigh is None:
                continue
            for a in mats:
                for b in neigh:
                    pairs_i.append(a)
                    pairs_j.append(b)
    return np.array(pairs_i, dtype=int), np.array(pairs_j, dtype=int)


def _components(n: int, pi: np.ndarray, pj: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """connected components labels of graph with n nodes and edges (pi[keep], pj[keep])"""
    graph = coo_matrix((np.ones(int(np.sum(keep))), (pi[keep], pj[keep])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels


# =============================================================================
# CLUSTERING
# =============================================================================

@dataclass
class ClusterResult:
    """clustering result

    clusters are sorted by decreasing size: cluster_id 0 is the largest
    cluster_assignment[matrix index] = cluster_id or -1 (not assigned, e.g. too small cluster)
    """
    clusters: List[Cluster]
    cluster_assignment: np.ndarray
    threshold: float = 0.0
    mode: str = ""
    symmetry: Any = 'cubic'
    mapdimension: Optional[Tuple[int, int]] = None
    _stats_cache: Dict = field(default_factory=dict, repr=False)

    def get_cluster(self, cluster_id: int) -> Optional[Cluster]:
        for cluster in self.clusters:
            if cluster.cluster_id == cluster_id:
                return cluster
        return None

    def get_cluster_images(self) -> Dict[int, List[int]]:
        """cluster_id -> sorted list of image indices"""
        return {c.cluster_id: sorted(set(c.image_indices)) for c in self.clusters}

    def get_image_clusters(self) -> Dict[int, List[int]]:
        """image_index -> sorted list of cluster_ids"""
        image_to_clusters = {}
        for cluster in self.clusters:
            for img in set(cluster.image_indices):
                image_to_clusters.setdefault(img, []).append(cluster.cluster_id)
        return {k: sorted(v) for k, v in image_to_clusters.items()}

    def get_cluster_positions_2d(self, mapdimension: Optional[Tuple[int, int]] = None
                                 ) -> Dict[int, List[Tuple[int, int]]]:
        """cluster_id -> list of (row, col) map positions"""
        _, ncols = self._mapdim(mapdimension)
        return {c.cluster_id: [divmod(img, ncols) for img in sorted(set(c.image_indices))]
                for c in self.clusters}

    def get_image_positions_2d(self, mapdimension: Optional[Tuple[int, int]] = None
                               ) -> Dict[int, Tuple[int, int]]:
        """image_index -> (row, col)"""
        _, ncols = self._mapdim(mapdimension)
        return {img: divmod(img, ncols) for img in self.get_image_clusters()}

    def _mapdim(self, mapdimension):
        mapdim = mapdimension if mapdimension is not None else self.mapdimension
        if mapdim is None:
            raise ValueError("mapdimension=(nrows, ncols) must be given")
        return mapdim

    def get_cluster_stats(self, matrices: Union[UBMatrixAnalyzer, List[UBMatrix]],
                          mapdimension: Optional[Tuple[int, int]] = None) -> List[ClusterStats]:
        """ClusterStats of all clusters (same order as self.clusters, i.e. by decreasing size)"""
        mapdim = tuple(self._mapdim(mapdimension))
        key = (id(matrices), mapdim)
        if key not in self._stats_cache:
            self._stats_cache[key] = [compute_cluster_stats(c, matrices, mapdim, self.symmetry)
                                      for c in self.clusters]
        return self._stats_cache[key]

    def get_cluster_stats_dict(self, matrices, mapdimension=None) -> Dict[int, ClusterStats]:
        """cluster_id -> ClusterStats  (use this rather than indexing a list with a cluster_id)"""
        return {s.cluster_id: s for s in self.get_cluster_stats(matrices, mapdimension)}

    def get_cluster_stats_by_image(self, matrices, image_index: int,
                                   mapdimension=None) -> List[ClusterStats]:
        """ClusterStats of clusters containing image_index"""
        return [s for s in self.get_cluster_stats(matrices, mapdimension) if s.contains_image(image_index)]

    def get_cluster_stats_sorted(self, matrices, sort_by: str = 'size', descending: bool = True,
                                 mapdimension=None) -> List[ClusterStats]:
        return sort_cluster_stats(self.get_cluster_stats(matrices, mapdimension), sort_by, descending)

    def __repr__(self):
        return (f"ClusterResult({len(self.clusters)} clusters, threshold={self.threshold}°, "
                f"symmetry={self.symmetry if isinstance(self.symmetry, str) or self.symmetry is None else 'custom'}, "
                f"mode={self.mode})")


def create_cluster_result(analyzer: UBMatrixAnalyzer, labels: np.ndarray, threshold: float,
                          mode: str, min_cluster_size: int = 1, symmetry='cubic',
                          mapdimension=None) -> ClusterResult:
    """build ClusterResult from labels (one per matrix, -1 = not assigned)

    cluster ids are renumbered by decreasing size; clusters smaller than min_cluster_size are dropped
    """
    labels = np.asarray(labels)
    uniq, counts = np.unique(labels[labels >= 0], return_counts=True)
    order = np.argsort(-counts, kind='stable')
    clusters = []
    assignment = np.full(len(analyzer.matrices), -1, dtype=int)
    for lab in uniq[order]:
        members = np.flatnonzero(labels == lab)
        if len(members) < min_cluster_size:
            continue
        cid = len(clusters)
        assignment[members] = cid
        clusters.append(Cluster(cluster_id=cid,
                                matrix_indices=members.tolist(),
                                image_indices=analyzer.image_indices[members].tolist(),
                                grain_indices=analyzer.grain_indices[members].tolist()))
    return ClusterResult(clusters=clusters, cluster_assignment=assignment, threshold=threshold,
                         mode=mode, symmetry=symmetry, mapdimension=mapdimension)


def _select(analyzer: UBMatrixAnalyzer, grain_index: Optional[int]) -> np.ndarray:
    if grain_index is None:
        return np.arange(len(analyzer.matrices))
    return np.flatnonzero(analyzer.grain_indices == grain_index)


def cluster_matrices_with_connectivity(analyzer: UBMatrixAnalyzer, threshold: float,
                                       mapdimension: Optional[Tuple[int, int]] = None,
                                       grain_index: Optional[int] = None,
                                       connectivity_type: Union[str, int] = '8',
                                       symmetry='cubic', min_cluster_size: int = 1) -> ClusterResult:
    """cluster matrices of neighbouring map points with misorientation < threshold (deg)

    only matrices of the same material are linked. Only neighbour pairs are computed: fast.
    """
    mapdim = analyzer.check_mapdimension(mapdimension)
    subset = _select(analyzer, grain_index)
    pi, pj = neighbour_pairs(analyzer, mapdim, connectivity_type, subset=subset)
    angles, _ = misorientation(analyzer.rotations[pi], analyzer.rotations[pj],
                               get_symmetry_operators(symmetry))
    keep = (angles < threshold) & (analyzer.materials[pi] == analyzer.materials[pj])
    labels = _components(len(analyzer.matrices), pi, pj, keep)
    mask = np.ones(len(labels), dtype=bool)
    mask[subset] = False
    labels[mask] = -1
    mode = f"{'all' if grain_index is None else f'grain_{grain_index}'}_connected_{connectivity_type}"
    return create_cluster_result(analyzer, labels, threshold, mode, min_cluster_size, symmetry, mapdim)


def cluster_matrices(analyzer: UBMatrixAnalyzer, threshold: float,
                     grain_index: Optional[int] = None, symmetry='cubic',
                     min_cluster_size: int = 1, mapdimension=None) -> ClusterResult:
    """cluster matrices by misorientation only (no spatial constraint), single linkage

    all pairs are computed (row by row): slower than cluster_matrices_with_connectivity()
    """
    subset = _select(analyzer, grain_index)
    rots = analyzer.rotations[subset]
    mats = analyzer.materials[subset]
    symops = get_symmetry_operators(symmetry)
    pi, pj = [], []
    for k in range(len(subset) - 1):
        others = np.arange(k + 1, len(subset))
        angles, _ = misorientation(np.broadcast_to(rots[k], (len(others), 3, 3)), rots[others], symops)
        close = others[(angles < threshold) & (mats[others] == mats[k])]
        pi.extend([k] * len(close))
        pj.extend(close.tolist())
    pi, pj = np.array(pi, dtype=int), np.array(pj, dtype=int)
    sub_labels = _components(len(subset), pi, pj, np.ones(len(pi), dtype=bool))
    labels = np.full(len(analyzer.matrices), -1, dtype=int)
    labels[subset] = sub_labels
    mode = 'all' if grain_index is None else f'grain_{grain_index}'
    return create_cluster_result(analyzer, labels, threshold, mode, min_cluster_size, symmetry,
                                 mapdimension if mapdimension is not None else analyzer.mapdimension)


def split_non_compact_clusters(result: ClusterResult, analyzer: UBMatrixAnalyzer,
                               mapdimension: Optional[Tuple[int, int]] = None,
                               min_cluster_size: int = 1, max_gap: int = 1) -> ClusterResult:
    """split each cluster into its spatially connected parts

    max_gap: map points of a cluster at Chebyshev distance <= max_gap are connected
             (1: touching including corners, 2: bridge a 1-point gap, ...)
    """
    mapdim = analyzer.check_mapdimension(mapdimension if mapdimension is not None else result.mapdimension)
    members = np.flatnonzero(result.cluster_assignment >= 0)
    pi, pj = neighbour_pairs(analyzer, mapdim, connectivity=max_gap, subset=members)
    keep = result.cluster_assignment[pi] == result.cluster_assignment[pj]
    labels = _components(len(analyzer.matrices), pi, pj, keep)
    labels[result.cluster_assignment < 0] = -1
    return create_cluster_result(analyzer, labels, result.threshold, f"{result.mode}_split",
                                 min_cluster_size, result.symmetry, mapdim)


def is_cluster_compact(cluster: Cluster, analyzer: UBMatrixAnalyzer,
                       mapdimension: Optional[Tuple[int, int]] = None, max_gap: int = 1) -> bool:
    """True if the map points of cluster form a single connected region (see split_non_compact_clusters)"""
    members = np.array(cluster.matrix_indices, dtype=int)
    if len(set(cluster.image_indices)) <= 1:
        return True
    pi, pj = neighbour_pairs(analyzer, mapdimension, connectivity=max_gap, subset=members)
    labels = _components(len(analyzer.matrices), pi, pj, np.ones(len(pi), dtype=bool))
    return len(np.unique(labels[members])) == 1


# =============================================================================
# SYMMETRY CHECK (ALARM)
# =============================================================================

def check_symmetry_equivalents(analyzer: UBMatrixAnalyzer,
                               mapdimension: Optional[Tuple[int, int]] = None,
                               symmetry='cubic', angle_tol: float = 2.0,
                               duplicate_tol: float = 0.05, distortion_tol: float = 0.05,
                               verbose: bool = True, nb_examples: int = 5) -> Dict[str, Any]:
    """detect UB matrices that look different but are equivalent by crystal symmetry

    compares matrices at the same or neighbouring (8-connectivity) map points:
    - 'sym_equivalent_pairs': raw angle > angle_tol but symmetry-reduced misorientation < angle_tol
      (the .fit matrices are NOT reduced to a common representative: harmless for the clustering
      of this module, which is symmetry-aware, but wrong for any raw element-wise comparison
      e.g. GT.getRotationAngleFrom2Matrices or maps of UB matrix elements)
    - 'same_image_duplicates': 2 grains of the same image with misorientation < duplicate_tol
      (same grain indexed twice, e.g. g0 and g500)
    - 'distorted_UB': UB whose singular values differ from 1 by more than distortion_tol
      (poor refinement: orientation is taken as the closest rotation)

    return dict of lists of (matrix index i, matrix index j, raw angle, reduced angle)
    """
    pi, pj = neighbour_pairs(analyzer, mapdimension, connectivity='8')
    same_mat = analyzer.materials[pi] == analyzer.materials[pj]
    pi, pj = pi[same_mat], pj[same_mat]
    raw, _ = misorientation(analyzer.rotations[pi], analyzer.rotations[pj], IDENTITY_OPS)
    reduced, _ = misorientation(analyzer.rotations[pi], analyzer.rotations[pj],
                                get_symmetry_operators(symmetry))

    sym_eq = np.flatnonzero((raw > angle_tol) & (reduced < angle_tol))
    same_img = analyzer.image_indices[pi] == analyzer.image_indices[pj]
    dupl = np.flatnonzero(same_img & (reduced < duplicate_tol))

    all_ub = np.array([m.matrix for m in analyzer.matrices]).reshape(-1, 3, 3)
    singvals = np.linalg.svd(all_ub, compute_uv=False)
    distorted = np.flatnonzero(np.abs(singvals - 1).max(axis=1) > distortion_tol)

    report = {
        'sym_equivalent_pairs': [(int(pi[k]), int(pj[k]), float(raw[k]), float(reduced[k])) for k in sym_eq],
        'same_image_duplicates': [(int(pi[k]), int(pj[k]), float(raw[k]), float(reduced[k])) for k in dupl],
        'distorted_UB': [(int(k), singvals[k].round(3).tolist()) for k in distorted],
        'nb_pairs_checked': len(pi),
    }

    if verbose:
        mats = analyzer.matrices
        def _name(k):
            return f"img {mats[k].image_index} g{mats[k].grain_index}"
        if len(sym_eq):
            GT.printred(f"WARNING: {len(sym_eq)} pairs of neighbouring UB matrices look different "
                        f"(raw angle > {angle_tol}°) but are equivalent by {symmetry} symmetry "
                        f"(misorientation < {angle_tol}°). Matrices are NOT reduced to a common "
                        f"representative: use symmetry-aware tools (this module) for any comparison.")
            for i, j, a, b in report['sym_equivalent_pairs'][:nb_examples]:
                print(f"    {_name(i)} <-> {_name(j)}: raw {a:.2f}°, reduced {b:.3f}°")
        else:
            GT.printgreen(f"OK: no neighbouring UB matrices equivalent by {symmetry} symmetry "
                          f"with raw angle > {angle_tol}° ({len(pi)} pairs checked)")
        if len(dupl):
            GT.printyellow(f"NOTE: {len(dupl)} pairs of grains in the same image have the same orientation "
                           f"(< {duplicate_tol}°): duplicated indexation")
            for i, j, _, b in report['same_image_duplicates'][:nb_examples]:
                print(f"    {_name(i)} <-> {_name(j)}: {b:.3f}°")
        if len(distorted):
            GT.printyellow(f"NOTE: {len(distorted)} UB matrices far from a rotation "
                           f"(singular values off by > {100 * distortion_tol:.0f}%): poor refinement?")
            for k, sv in report['distorted_UB'][:nb_examples]:
                print(f"    {_name(k)}: singular values {sv}")
    return report


def align_to_cluster_reference(result: ClusterResult, analyzer: UBMatrixAnalyzer,
                               mapdimension=None) -> np.ndarray:
    """return array (N, 3, 3) of UB matrices transformed (UB.S) to the symmetry variant closest
    to the reference matrix of their cluster. Unassigned matrices are returned unchanged.

    useful before plotting matrix-derived quantities (Euler angles, raw element maps, ...)
    """
    symops = get_symmetry_operators(result.symmetry)
    aligned = np.array([m.matrix for m in analyzer.matrices]).reshape(-1, 3, 3).copy()
    for stats in result.get_cluster_stats(analyzer, mapdimension):
        idx = np.array(stats.matrix_indices)
        ref = analyzer.rotations[stats.reference_matrix_index]
        _, best = misorientation(np.broadcast_to(ref, (len(idx), 3, 3)), analyzer.rotations[idx], symops)
        aligned[idx] = aligned[idx] @ symops[best]
    return aligned


# =============================================================================
# STATISTICS
# =============================================================================

def compute_cluster_stats(cluster: Cluster, analyzer: UBMatrixAnalyzer,
                          mapdimension: Tuple[int, int], symmetry='cubic') -> ClusterStats:
    """statistics of a cluster: reference = matrix closest to the barycenter of the cluster"""
    _, ncols = analyzer.check_mapdimension(mapdimension)
    idx = np.array(cluster.matrix_indices, dtype=int)
    rows, cols = np.divmod(analyzer.image_indices[idx], ncols)
    mean_row, mean_col = float(rows.mean()), float(cols.mean())

    k_ref = int(np.argmin((rows - mean_row) ** 2 + (cols - mean_col) ** 2))
    ref_idx = int(idx[k_ref])
    ref_rot = analyzer.rotations[ref_idx]

    symops = get_symmetry_operators(symmetry)
    angles, best = misorientation(np.broadcast_to(ref_rot, (len(idx), 3, 3)), analyzer.rotations[idx], symops)
    # mean orientation from symmetry-aligned rotations (valid for small spreads)
    mean_orientation = orthonormalize(np.mean(analyzer.rotations[idx] @ symops[best], axis=0))

    return ClusterStats(
        cluster_id=cluster.cluster_id,
        size=cluster.size,
        nb_pixels=cluster.nb_pixels,
        reference_matrix=analyzer.matrices[ref_idx].matrix,
        reference_matrix_index=ref_idx,
        reference_image_index=int(analyzer.image_indices[ref_idx]),
        mean_orientation=mean_orientation,
        mean_position=(mean_row, mean_col),
        min_position=(float(rows.min()), float(cols.min())),
        max_position=(float(rows.max()), float(cols.max())),
        std_orientation=float(np.std(angles)),
        mean_angular_deviation=float(np.mean(angles)),
        max_angular_deviation=float(np.max(angles)),
        image_indices=cluster.image_indices,
        matrix_indices=cluster.matrix_indices,
        grain_indices=cluster.grain_indices,
        materials=sorted(set(analyzer.materials[idx].tolist())),
        all_angles=angles.tolist(),
    )


def sort_cluster_stats(cluster_stats: List[ClusterStats], sort_by: str = 'size',
                       descending: bool = True) -> List[ClusterStats]:
    """sort by 'size', 'nb_pixels', 'std_orientation', 'max_angular_deviation', 'width', 'height',
    'aspect_ratio', 'cluster_id', 'mean_row', 'mean_col'"""
    sort_keys = {
        'size': lambda x: x.size,
        'nb_pixels': lambda x: x.nb_pixels,
        'std_orientation': lambda x: x.std_orientation,
        'max_angular_deviation': lambda x: x.max_angular_deviation,
        'width': lambda x: x.width(),
        'height': lambda x: x.height(),
        'aspect_ratio': lambda x: x.aspect_ratio(),
        'cluster_id': lambda x: x.cluster_id,
        'mean_row': lambda x: x.mean_position[0],
        'mean_col': lambda x: x.mean_position[1],
    }
    if sort_by not in sort_keys:
        raise ValueError(f"Invalid sort_by: {sort_by}. Valid options: {list(sort_keys)}")
    return sorted(cluster_stats, key=sort_keys[sort_by], reverse=descending)


def get_angular_deviation_distribution(stats: ClusterStats) -> Dict[str, float]:
    """mean, std, min, max, median, q25, q75 of misorientations to the cluster reference"""
    a = np.array(stats.all_angles) if stats.all_angles else np.zeros(1)
    return {'mean': float(a.mean()), 'std': float(a.std()), 'min': float(a.min()), 'max': float(a.max()),
            'median': float(np.median(a)), 'q25': float(np.percentile(a, 25)),
            'q75': float(np.percentile(a, 75))}


def print_cluster_summary(result: ClusterResult, analyzer: UBMatrixAnalyzer,
                          mapdimension=None, nb_max: int = 30):
    """print one line per cluster (largest first)"""
    stats = result.get_cluster_stats(analyzer, mapdimension)
    nb_assigned = int(np.sum(result.cluster_assignment >= 0))
    print(f"{result}: {nb_assigned}/{len(result.cluster_assignment)} matrices assigned")
    for s in stats[:nb_max]:
        row, col = s.mean_position
        print(f"  Cluster {s.cluster_id:3d}: {s.size:5d} matrices, {s.nb_pixels:5d} pixels, "
              f"mean (row, col)=({row:5.1f}, {col:5.1f}), ref img {s.reference_image_index:5d}, "
              f"std={s.std_orientation:.2f}°, max dev={s.max_angular_deviation:.2f}°, grains {sorted(set(s.grain_indices))}")
    if len(stats) > nb_max:
        print(f"  ... and {len(stats) - nb_max} more clusters")


def get_reference_matrices(result: ClusterResult, analyzer: UBMatrixAnalyzer, min_size: int = 1,
                           which: str = 'reference', as_list: bool = True, decimals: int = 9,
                           verbose: bool = True) -> Dict[int, Any]:
    """reference orientation matrix of each cluster (largest clusters first)

    which: 'reference': UB matrix as written in the .fit file of the point closest to the cluster
                        barycenter (includes the refined deviatoric strain)
           'mean': mean orientation of the cluster (pure rotation, symmetry-aligned average)
    as_list: nested python lists (copy-paste ready, e.g. for LaueTools GUIs), else numpy arrays
    verbose: print one copy-paste ready line per cluster

    e.g. matrices for indexation of other images (GuessedUBMatrix / previousResults):
        refs = OC.get_reference_matrices(result, analyzer, min_size=30)
        UBlist = [np.array(m) for m in refs.values()]
    """
    matrices = {}
    for s in result.get_cluster_stats(analyzer):
        if s.size < min_size:
            continue
        if which == 'reference':
            mat = np.asarray(s.reference_matrix, dtype=float)
        elif which == 'mean':
            mat = np.asarray(s.mean_orientation, dtype=float)
        else:
            raise ValueError("which must be 'reference' or 'mean'")
        matrices[s.cluster_id] = np.round(mat, decimals).tolist() if as_list else mat
        if verbose:
            print(f"# cluster {s.cluster_id} ({s.nb_pixels} pts, ref image {s.reference_image_index})\n"
                  f"{format_matrix_list(mat, decimals)}")
    return matrices


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def guess_symmetry(analyzer: UBMatrixAnalyzer, dictmaterials=None):
    """'cubic' if all materials of the .fit files are cubic in LaueTools materials dict, else None"""
    from . import CrystalParameters as CP
    if dictmaterials is None:
        dictmaterials = DictLT.dict_Materials
    materials = sorted(set(analyzer.materials.tolist()))
    try:
        cubic = [CP.hasCubicSymmetry(mat, dictmaterials) for mat in materials]
    except KeyError as err:
        GT.printyellow(f"material {err} unknown: symmetry not taken into account (raw angles)")
        return None
    symmetry = 'cubic' if cubic and all(cubic) else None
    print(f"materials {materials} -> symmetry: {symmetry}")
    return symmetry


def analyze_ub_matrices(input_dir: Union[str, Path], threshold: float,
                        mapdimension: Tuple[int, int], prefix: str = '',
                        symmetry='cubic', grain_index: Optional[int] = None,
                        grain_indices: Optional[List[int]] = None,
                        spatial_connectivity: bool = True, connectivity_type: Union[str, int] = '8',
                        min_cluster_size: int = 1, check_symmetry: bool = True,
                        verbose: bool = False, plot_results: bool = False, figsize=(8, 7),
                        nbfiles_per_folder: Optional[int] = None, subfolder_prefix: str = 'images_'
                        ) -> Tuple[Union[ClusterResult, Dict[int, ClusterResult]], UBMatrixAnalyzer]:
    """load UB matrices of .fit files in input_dir and segment the map into clusters

    threshold: max misorientation (deg) between linked matrices
    mapdimension: (nrows, ncols) = (slow axis, fast axis) = invmapdims
    prefix: .fit file prefix, e.g. 'eiger4m_'
    symmetry: 'cubic', None (raw angles), 'auto' (cubic if all materials are cubic, see guess_symmetry())
              or array of symmetry operators
    grain_index: cluster only matrices of this grain index (from _g#.fit)
    grain_indices: list of grain indices, each clustered separately -> dict of results
    spatial_connectivity: if True link only same/neighbouring map points (recommended)
    connectivity_type: '4' or '8'
    min_cluster_size: drop clusters with less matrices
    check_symmetry: run check_symmetry_equivalents() and print warnings
    nbfiles_per_folder, subfolder_prefix: .fit files in subfolders of input_dir (see UBMatrixAnalyzer)

    return (result or dict of results, analyzer)
    """
    analyzer = UBMatrixAnalyzer(input_dir=input_dir, prefix=prefix, mapdimension=tuple(mapdimension),
                                nbfiles_per_folder=nbfiles_per_folder, subfolder_prefix=subfolder_prefix)
    n_loaded = analyzer.load_matrices(verbose=int(verbose))
    if n_loaded == 0:
        raise ValueError(f"No UB matrix found in {input_dir} with prefix '{prefix}'")
    analyzer.check_mapdimension()
    if verbose:
        print(f"Loaded {n_loaded} matrices from {len(analyzer.image_to_matrices)} images, "
              f"grain indices {analyzer.get_all_grain_indices()}, materials {sorted(set(analyzer.materials))}")
    if isinstance(symmetry, str) and symmetry == 'auto':
        symmetry = guess_symmetry(analyzer)
    if check_symmetry and symmetry is not None:
        check_symmetry_equivalents(analyzer, symmetry=symmetry)
        if isinstance(symmetry, str) and symmetry == 'cubic':
            check_crystal_frame_settings(analyzer)

    def _run(g_index):
        if spatial_connectivity:
            return cluster_matrices_with_connectivity(analyzer, threshold, grain_index=g_index,
                                                      connectivity_type=connectivity_type,
                                                      symmetry=symmetry, min_cluster_size=min_cluster_size)
        return cluster_matrices(analyzer, threshold, grain_index=g_index, symmetry=symmetry,
                                min_cluster_size=min_cluster_size)

    if grain_indices is not None:
        results = {g: _run(g) for g in grain_indices}
    else:
        results = {None: _run(grain_index)}

    for res in results.values():
        if verbose:
            print_cluster_summary(res, analyzer)
        if plot_results and res.clusters:
            plot_cluster_map(res, analyzer, figsize=figsize)

    if grain_indices is not None:
        return results, analyzer
    return results[None], analyzer


# =============================================================================
# MAPS
# =============================================================================

def primary_matrix_map(analyzer: UBMatrixAnalyzer, result: Optional[ClusterResult] = None,
                       mapdimension=None) -> np.ndarray:
    """2D map (nrows, ncols) of the index of the 'primary' matrix of each map point (-1 if none)

    primary = matrix with the largest nb of indexed spots at that point
    (among matrices assigned to a cluster if result is given)
    """
    nrows, ncols = analyzer.check_mapdimension(mapdimension)
    pmap = np.full(nrows * ncols, -1, dtype=int)
    best = np.full(nrows * ncols, -1, dtype=int)
    for idx in range(len(analyzer.matrices)):
        if result is not None and result.cluster_assignment[idx] < 0:
            continue
        img = analyzer.image_indices[idx]
        if analyzer.nb_indexed[idx] > best[img]:
            best[img] = analyzer.nb_indexed[idx]
            pmap[img] = idx
    return pmap.reshape(nrows, ncols)


def cluster_label_map(result: ClusterResult, analyzer: UBMatrixAnalyzer,
                      mapdimension=None, min_size: int = 1) -> np.ndarray:
    """2D map of cluster id of the primary matrix of each point (-1: none or cluster < min_size)"""
    pmap = primary_matrix_map(analyzer, result, mapdimension)
    labels = np.where(pmap >= 0, result.cluster_assignment[np.maximum(pmap, 0)], -1)
    if min_size > 1:
        small = [c.cluster_id for c in result.clusters if c.size < min_size]
        labels[np.isin(labels, small)] = -1
    return labels


def cluster_matrix_map(result: ClusterResult, analyzer: UBMatrixAnalyzer, cluster_id: int,
                       mapdimension=None) -> np.ndarray:
    """2D map of the matrix index of cluster_id at each map point (-1 elsewhere)

    uses ALL matrices of the cluster, also where the cluster is not the primary grain of the point.
    Several matrices of the cluster at one point (duplicated indexation): most indexed spots is kept
    """
    nrows, ncols = analyzer.check_mapdimension(mapdimension if mapdimension is not None else result.mapdimension)
    cluster = result.get_cluster(cluster_id)
    if cluster is None:
        raise ValueError(f"Cluster {cluster_id} not found")
    cmap = np.full(nrows * ncols, -1, dtype=int)
    best = np.full(nrows * ncols, -1, dtype=int)
    for idx in cluster.matrix_indices:
        img = analyzer.image_indices[idx]
        if analyzer.nb_indexed[idx] > best[img]:
            best[img] = analyzer.nb_indexed[idx]
            cmap[img] = idx
    return cmap.reshape(nrows, ncols)


def cluster_reference_rotation(result: ClusterResult, analyzer: UBMatrixAnalyzer, cluster_id: int,
                               reference: Union[None, str, int, np.ndarray] = None,
                               mapdimension=None) -> np.ndarray:
    """reference orientation (rotation matrix) of a cluster

    reference: None: matrix of the point closest to the cluster barycenter (default)
               'mean': mean orientation of the cluster
               int: image index (must belong to the cluster)
               (3, 3) array: any UB matrix
    """
    stats = result.get_cluster_stats_dict(analyzer, mapdimension)[cluster_id]
    if reference is None:
        return analyzer.rotations[stats.reference_matrix_index]
    if isinstance(reference, str):
        if reference == 'mean':
            return stats.mean_orientation
        raise ValueError(f"unknown reference '{reference}': use None, 'mean', an image index or a UB matrix")
    if np.ndim(reference) == 0:
        members = [i for i in stats.matrix_indices if analyzer.image_indices[i] == int(reference)]
        if not members:
            raise ValueError(f"image {reference} does not belong to cluster {cluster_id}")
        return analyzer.rotations[max(members, key=lambda i: analyzer.nb_indexed[i])]
    return orthonormalize(np.asarray(reference, dtype=float).reshape(3, 3))


def compute_grod_map(result: ClusterResult, analyzer: UBMatrixAnalyzer, mapdimension=None,
                     cluster_id: Optional[int] = None,
                     reference: Union[None, str, int, np.ndarray] = None) -> np.ndarray:
    """Grain Reference Orientation Deviation map (deg), NaN where not defined

    cluster_id=None: whole map, primary matrix of each point vs the reference of its own cluster
    cluster_id=k: only cluster k (all its points, also where it is not the primary grain),
                  vs its reference, chosen with reference (see cluster_reference_rotation())
    """
    symops = get_symmetry_operators(result.symmetry)
    if cluster_id is not None:
        cmap = cluster_matrix_map(result, analyzer, cluster_id, mapdimension)
        grod = np.full(cmap.shape, np.nan)
        valid = cmap >= 0
        ref = cluster_reference_rotation(result, analyzer, cluster_id, reference, mapdimension)
        grod[valid], _ = misorientation(np.broadcast_to(ref, (int(valid.sum()), 3, 3)),
                                        analyzer.rotations[cmap[valid]], symops)
        return grod

    pmap = primary_matrix_map(analyzer, result, mapdimension)
    grod = np.full(pmap.shape, np.nan)
    stats = result.get_cluster_stats_dict(analyzer, mapdimension)
    valid = pmap >= 0
    idx = pmap[valid]
    refs = np.array([analyzer.rotations[stats[c].reference_matrix_index]
                     for c in result.cluster_assignment[idx]]).reshape(-1, 3, 3)
    grod[valid], _ = misorientation(refs, analyzer.rotations[idx], symops)
    return grod


def compute_kam(analyzer: UBMatrixAnalyzer, result: Optional[ClusterResult] = None,
                mapdimension=None, connectivity: Union[str, int] = '8',
                max_angle: float = 2.0, symmetry='cubic',
                cluster_id: Optional[int] = None) -> np.ndarray:
    """Kernel Average Misorientation map (deg)

    for each map point: mean misorientation between its primary matrix and those of its
    neighbours, ignoring neighbours with misorientation > max_angle (grain boundaries)
    and, if result is given, neighbours belonging to another cluster.
    connectivity: '4', '8' (first neighbours) or int d (all points within Chebyshev distance d)
    cluster_id: KAM of this cluster only (all its points, also where it is not the primary grain)
    return (nrows, ncols) array, NaN where no valid neighbour
    """
    nrows, ncols = analyzer.check_mapdimension(mapdimension)
    if result is not None:
        symmetry = result.symmetry
    if cluster_id is not None:
        if result is None:
            raise ValueError("cluster_id needs result")
        pmap = cluster_matrix_map(result, analyzer, cluster_id, (nrows, ncols))
    else:
        pmap = primary_matrix_map(analyzer, result, (nrows, ncols))
    symops = get_symmetry_operators(symmetry)
    total = np.zeros((nrows, ncols))
    count = np.zeros((nrows, ncols))
    rows, cols = np.nonzero(pmap >= 0)
    for dr, dc in _forward_offsets(connectivity):
        nr, nc = rows + dr, cols + dc
        inside = (nr >= 0) & (nr < nrows) & (nc >= 0) & (nc < ncols)
        r0, c0, r1, c1 = rows[inside], cols[inside], nr[inside], nc[inside]
        i0, i1 = pmap[r0, c0], pmap[r1, c1]
        ok = i1 >= 0
        if result is not None:
            ok &= result.cluster_assignment[i0] == result.cluster_assignment[np.maximum(i1, 0)]
        ok &= analyzer.materials[i0] == analyzer.materials[np.maximum(i1, 0)]
        r0, c0, r1, c1, i0, i1 = r0[ok], c0[ok], r1[ok], c1[ok], i0[ok], i1[ok]
        angles, _ = misorientation(analyzer.rotations[i0], analyzer.rotations[i1], symops)
        small = angles <= max_angle
        # each pair contributes to both points
        np.add.at(total, (r0[small], c0[small]), angles[small])
        np.add.at(count, (r0[small], c0[small]), 1)
        np.add.at(total, (r1[small], c1[small]), angles[small])
        np.add.at(count, (r1[small], c1[small]), 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(count > 0, total / count, np.nan)


# =============================================================================
# GRAIN PROPERTIES FROM .fit FILES (strain, lattice parameters, Euler angles)
# =============================================================================

STRAIN_COMPONENTS = {'xx': (0, 0), 'yy': (1, 1), 'zz': (2, 2), 'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
LATTICE_NAMES = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
EULER_NAMES = ['phi', 'theta', 'psi']
PROPERTY_GROUPS = {
    'orientation': ['grod'] + EULER_NAMES,
    'strain sample frame': [f'e{c}_sample' for c in STRAIN_COMPONENTS],
    'strain crystal frame': [f'e{c}' for c in STRAIN_COMPONENTS],
    'lattice parameters': LATTICE_NAMES,
    'quality': ['nb_indexed', 'pixdev'],
}
#: properties expressed in crystal axes (depend on the cubic setting of the UB matrix)
CRYSTAL_FRAME_PROPERTIES = set([f'e{c}' for c in STRAIN_COMPONENTS] + LATTICE_NAMES + EULER_NAMES)


def is_strain_property(prop: str) -> bool:
    return prop.startswith('e') and prop[1:3] in STRAIN_COMPONENTS


def property_label(prop: str, strain_multiplier: float = 1e4) -> str:
    """axis / colorbar label of a property"""
    if prop == 'grod':
        return 'misorientation to reference (deg)'
    if is_strain_property(prop):
        frame = 'sample' if prop.endswith('_sample') else 'crystal'
        return f'ε{prop[1:3]} {frame} (x{1 / strain_multiplier:.0E})'
    if prop in ('a', 'b', 'c'):
        return f'{prop} (Angst)'
    if prop in ('alpha', 'beta', 'gamma') or prop in EULER_NAMES:
        return f'{prop} (deg)'
    return {'nb_indexed': 'nb of indexed spots', 'pixdev': 'mean pixel deviation'}.get(prop, prop)


def _matrix_props(analyzer: UBMatrixAnalyzer, idx: int) -> Dict[str, Any]:
    """quantities of the grain block of matrix idx in its own .fit file (read again from the file
    if the analyzer was loaded without them)"""
    m = analyzer.matrices[idx]
    if not m.props:
        grains = read_fitfile_grains(analyzer.input_dir / m.filename)
        k = m.source_grain if m.source_grain is not None else 0
        m.props = grains[k]['props'] if k < len(grains) else {}
    return m.props


def _lattice_to_metric(p: np.ndarray) -> np.ndarray:
    a, b, c = p[:3]
    ca, cb, cg = np.cos(np.radians(p[3:6]))
    return np.array([[a * a, a * b * cg, a * c * cb], [a * b * cg, b * b, b * c * ca], [a * c * cb, b * c * ca, c * c]])


def _metric_to_lattice(G: np.ndarray) -> np.ndarray:
    a, b, c = np.sqrt(np.diag(G))
    return np.array([a, b, c, np.degrees(np.arccos(G[1, 2] / (b * c))),
                     np.degrees(np.arccos(G[0, 2] / (a * c))), np.degrees(np.arccos(G[0, 1] / (a * b)))])


def sample_to_crystal_strain(strain_sample: np.ndarray, UB: np.ndarray, sampletilt: float = 40.0) -> np.ndarray:
    """crystal-frame strain in the axes of UB from sample-frame strain
    (inverse of CrystalParameters.strain_from_crystal_to_sample_frame2)"""
    P = GT.matRot([0, 1, 0], -sampletilt)
    # pure rotation (polar decomposition of UB), as in strain_from_crystal_to_sample_frame2
    uu, _, vvt = np.linalg.svd(np.asarray(UB, dtype=np.float64))
    M = P.T @ uu @ vvt
    eps = M.T @ strain_sample @ M
    return 0.5 * (eps + eps.T)


def file_setting_operator(analyzer: UBMatrixAnalyzer, idx: int,
                          sampletilt: float = 40.0) -> Tuple[Optional[np.ndarray], float]:
    """cubic operator S relating the crystal axes of the quantities written in the .fit file of matrix idx
    (crystal-frame strain, lattice parameters) to the axes of the UB matrix written in the same file:
        strain in UB axes = S^T . strain_file . S   (same for the metric tensor)

    S differs from identity when LaueTools wrote the UB in its 'lowest Euler angles' cubic representative
    (choose_UB_MinEulerepresentative) while strain and lattice parameters stayed in the refinement setting.
    S is found by comparing with the crystal strain recomputed from the (setting independent)
    sample-frame strain. return S and residual (max abs difference, absolute strain; file rounding ~1e-5)
    """
    props = _matrix_props(analyzer, idx)
    if 'strain_crystal' not in props or 'strain_sample' not in props:
        return None, np.nan
    eps_ub = sample_to_crystal_strain(props['strain_sample'], analyzer.matrices[idx].matrix, sampletilt)
    candidates = np.einsum('sji,jk,skl->sil', CUBIC_ROTATIONS, props['strain_crystal'], CUBIC_ROTATIONS)
    residuals = np.abs(candidates - eps_ub).max(axis=(1, 2))
    k = int(np.argmin(residuals))
    return CUBIC_ROTATIONS[k], float(residuals[k])


def matrix_properties(analyzer: UBMatrixAnalyzer, idx: int, S_align: Optional[np.ndarray] = None,
                      cubic: bool = True, sampletilt: float = 40.0,
                      setting_tol: float = 5e-5) -> Dict[str, float]:
    """strain components, lattice parameters, Euler angles, quality of matrix idx, read from its own
    .fit file (grain block of this matrix)

    crystal-frame quantities are expressed in the crystal axes of UB.S_align (S_align: cubic operator
    bringing the matrix to the symmetry variant of its cluster reference, None: axes of the written UB):
    - sample-frame strain: as written (independent of the setting)
    - crystal-frame strain: recomputed from sample-frame strain in the axes of UB.S_align (cubic);
      as written if not cubic
    - lattice parameters: written values transformed to the axes of UB.S_align (see file_setting_operator()),
      lengths rescaled so that a keeps the written (fixed) value of the deviatoric refinement;
      NaN if the file setting cannot be determined (residual > setting_tol)
    - Euler angles: LaueTools convention (orientations.calc_Euler_angles) of UB.S_align.B0
    """
    from . import orientations as ORI
    key = (idx, None if S_align is None else tuple(np.asarray(S_align).ravel()), cubic, sampletilt)
    cache = analyzer.__dict__.setdefault('_prop_cache', {})
    if key in cache:
        return cache[key]
    m = analyzer.matrices[idx]
    props = _matrix_props(analyzer, idx)
    S = np.eye(3) if S_align is None else np.asarray(S_align, dtype=float)
    UBa = m.matrix @ S
    out = {'nb_indexed': float(m.nb_indexed), 'pixdev': np.nan if m.pixdev is None else float(m.pixdev)}

    es = props.get('strain_sample')
    ec = None
    if cubic and es is not None:
        ec = sample_to_crystal_strain(es, UBa, sampletilt)
    elif not cubic:
        ec = props.get('strain_crystal')
    for comp, (i, j) in STRAIN_COMPONENTS.items():
        out[f'e{comp}_sample'] = es[i, j] if es is not None else np.nan
        out[f'e{comp}'] = ec[i, j] if ec is not None else np.nan

    lattice = np.full(6, np.nan)
    lp = props.get('latticeparams')
    if lp is not None and len(lp) == 6:
        if not cubic:
            lattice = np.asarray(lp, dtype=float)
        else:
            S_file, residual = file_setting_operator(analyzer, idx, sampletilt)
            if S_file is not None and residual < setting_tol:
                G = S.T @ (S_file.T @ _lattice_to_metric(lp) @ S_file) @ S
                lattice = _metric_to_lattice(G)
                lattice[:3] *= lp[0] / lattice[0]
    out.update(dict(zip(LATTICE_NAMES, lattice)))

    B0 = props.get('B0', np.eye(3))
    euler = ORI.calc_Euler_angles(UBa @ B0)
    out.update(dict(zip(EULER_NAMES, np.full(3, np.nan) if euler is None else euler)))
    cache[key] = out
    return out


def check_crystal_frame_settings(analyzer: UBMatrixAnalyzer, sampletilt: float = 40.0,
                                 setting_tol: float = 5e-5, verbose: bool = True) -> Dict[str, Any]:
    """check if crystal-frame strain and lattice parameters of .fit files are expressed in the axes of the
    written UB matrix (cubic only). return dict: 'consistent', 'permuted', 'unresolved' lists of matrix indices"""
    report = {'consistent': [], 'permuted': [], 'unresolved': [], 'no_strain': []}
    for idx in range(len(analyzer.matrices)):
        S_file, residual = file_setting_operator(analyzer, idx, sampletilt)
        if S_file is None:
            report['no_strain'].append(idx)
        elif residual >= setting_tol:
            report['unresolved'].append(idx)
        elif np.allclose(S_file, np.eye(3)):
            report['consistent'].append(idx)
        else:
            report['permuted'].append(idx)
    if verbose:
        nb = len(analyzer.matrices)
        if report['permuted']:
            GT.printred(f"WARNING: in {len(report['permuted'])}/{nb} grain blocks of .fit files, crystal-frame strain "
                        "and lattice parameters are expressed in a cubic setting different from the written UB "
                        "matrix (UB written as 'lowest Euler angles' representative). Maps of crystal-frame strain "
                        "or lattice parameters read directly from files (e.g. ffs0.exx, ffs0.a) mix settings. "
                        "Sample-frame strain is not affected. Use matrix_properties() / cluster_property_map().")
        else:
            GT.printgreen(f"OK: crystal-frame strain and lattice parameters consistent with written UB "
                          f"({len(report['consistent'])} grain blocks)")
        if report['unresolved']:
            GT.printyellow(f"NOTE: {len(report['unresolved'])} grain blocks with unresolved setting "
                           f"(lattice parameters set to NaN)")
    return report


def _alignment_operators(result: ClusterResult, analyzer: UBMatrixAnalyzer, cluster_id: int,
                         indices: np.ndarray, mapdimension=None) -> List[Optional[np.ndarray]]:
    """cubic operators bringing matrices indices to the symmetry variant of the cluster reference"""
    if not (isinstance(result.symmetry, str) and result.symmetry == 'cubic'):
        return [None] * len(indices)
    ref = analyzer.rotations[result.get_cluster_stats_dict(analyzer, mapdimension)[cluster_id].reference_matrix_index]
    _, best = misorientation(np.broadcast_to(ref, (len(indices), 3, 3)), analyzer.rotations[indices], CUBIC_ROTATIONS)
    return [CUBIC_ROTATIONS[k] for k in best]


def cluster_property_map(result: ClusterResult, analyzer: UBMatrixAnalyzer, cluster_id: int, prop: str,
                         aligned: bool = True, mapdimension=None, sampletilt: float = 40.0) -> np.ndarray:
    """2D map of a property of cluster_id (NaN elsewhere), each point read from the .fit file (and grain
    block) of the matrix of this cluster at this point

    prop: 'grod', 'phi', 'theta', 'psi', 'exx_sample' ... 'eyz_sample', 'exx' ... 'eyz' (crystal frame),
          'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'nb_indexed', 'pixdev'  (see PROPERTY_GROUPS)
    aligned: crystal-frame quantities in the axes of the cluster reference matrix (cubic), so that they
             are comparable from point to point. False: axes of the UB written in each file
    strain values are absolute (not multiplied)
    """
    if prop == 'grod':
        return compute_grod_map(result, analyzer, mapdimension, cluster_id=cluster_id)
    cmap = cluster_matrix_map(result, analyzer, cluster_id, mapdimension)
    values = np.full(cmap.shape, np.nan)
    rows, cols = np.nonzero(cmap >= 0)
    indices = cmap[rows, cols]
    cubic = isinstance(result.symmetry, str) and result.symmetry == 'cubic'
    ops = _alignment_operators(result, analyzer, cluster_id, indices, mapdimension) if aligned else [None] * len(indices)
    for r, c, idx, S in zip(rows, cols, indices, ops):
        values[r, c] = matrix_properties(analyzer, int(idx), S, cubic, sampletilt)[prop]
    return values


def property_map(result: ClusterResult, analyzer: UBMatrixAnalyzer, prop: str, aligned: bool = True,
                 mapdimension=None, sampletilt: float = 40.0) -> np.ndarray:
    """2D map of a property of the primary grain of each point, crystal-frame quantities aligned to the
    reference of the cluster of each point (NaN for points not assigned to a cluster)"""
    if prop == 'grod':
        return compute_grod_map(result, analyzer, mapdimension)
    pmap = primary_matrix_map(analyzer, result, mapdimension)
    values = np.full(pmap.shape, np.nan)
    cubic = isinstance(result.symmetry, str) and result.symmetry == 'cubic'
    for cid in np.unique(result.cluster_assignment[pmap[pmap >= 0]]):
        rows, cols = np.nonzero((pmap >= 0) & (result.cluster_assignment[np.maximum(pmap, 0)] == cid))
        indices = pmap[rows, cols]
        ops = _alignment_operators(result, analyzer, int(cid), indices, mapdimension) if aligned else [None] * len(indices)
        for r, c, idx, S in zip(rows, cols, indices, ops):
            values[r, c] = matrix_properties(analyzer, int(idx), S, cubic, sampletilt)[prop]
    return values


# =============================================================================
# CLUSTER ENVELOPES AND BOUNDARIES
# =============================================================================

def _disk(radius: float) -> np.ndarray:
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return xx ** 2 + yy ** 2 <= radius ** 2 + 1e-9


def cluster_mask(result: ClusterResult, analyzer: UBMatrixAnalyzer, cluster_id: int,
                 mapdimension=None) -> np.ndarray:
    """boolean 2D map of the points of cluster_id (all its matrices, not only primary grains)"""
    return cluster_matrix_map(result, analyzer, cluster_id, mapdimension) >= 0


def cluster_envelope(result: ClusterResult, analyzer: UBMatrixAnalyzer, cluster_id: int,
                     method: str = 'closing', radius: float = 2.0, mapdimension=None,
                     min_component_points: int = 3) -> Dict[str, Any]:
    """envelope of a (sparse) cluster, i.e. the region of the map occupied by the grain

    method: 'closing': morphological closing with a disk of radius (map steps) + hole filling.
                       Bridges gaps up to ~2*radius, follows concave shapes (recommended)
            'hull': convex hull of the points (extreme limits, overestimates concave grains)
            'bbox': bounding box
    min_component_points: envelope pieces containing less cluster points are removed (stray points)
    return dict: 'mask' (bool 2D envelope), 'points' (bool 2D cluster points), 'area' (nb of map
    points in envelope), 'nb_points', 'nb_outliers' (points outside envelope),
    'fill_fraction' (points in envelope / area: sparsity), 'bbox'
    (row_min, row_max, col_min, col_max), 'centroid' (row, col) of envelope,
    'extent' (max distance between 2 points of the cluster, map steps)
    """
    from scipy import ndimage as ndi
    points = cluster_mask(result, analyzer, cluster_id, mapdimension)
    nrows, ncols = points.shape
    rows, cols = np.nonzero(points)
    if method == 'closing':
        pad = int(np.ceil(radius)) + 1
        padded = np.pad(points, pad)
        closed = ndi.binary_closing(padded, structure=_disk(radius))[pad:-pad, pad:-pad]
        env = ndi.binary_fill_holes(closed | points)
    elif method == 'hull':
        from scipy.spatial import ConvexHull, QhullError
        from matplotlib.path import Path as MplPath
        corners = np.concatenate([np.column_stack([cols + dx, rows + dy])
                                  for dx in (-0.5, 0.5) for dy in (-0.5, 0.5)])
        try:
            hull = ConvexHull(corners)
            poly = MplPath(corners[hull.vertices])
            yy, xx = np.mgrid[0:nrows, 0:ncols]
            env = poly.contains_points(np.column_stack([xx.ravel(), yy.ravel()])).reshape(nrows, ncols) | points
        except QhullError:  # aligned points
            env = points.copy()
    elif method == 'bbox':
        env = np.zeros_like(points)
        env[rows.min():rows.max() + 1, cols.min():cols.max() + 1] = True
    else:
        raise ValueError("method must be 'closing', 'hull' or 'bbox'")

    if min_component_points > 1:
        comp, nb_comp = ndi.label(env, structure=np.ones((3, 3), dtype=bool))
        counts = np.bincount(comp[points], minlength=nb_comp + 1)
        keep = counts >= min_component_points
        keep[0] = False
        if keep.any():  # never remove everything
            env = keep[comp]

    erows, ecols = np.nonzero(env)
    # extent: max distance between cluster points (computed on convex hull vertices would be faster)
    pts = np.column_stack([rows, cols]).astype(float)
    if len(pts) > 1:
        from scipy.spatial.distance import pdist
        extent = float(pdist(pts).max())
    else:
        extent = 0.0
    return {'cluster_id': cluster_id, 'method': method, 'radius': radius,
            'mask': env, 'points': points, 'area': int(env.sum()), 'nb_points': int(points.sum()),
            'nb_outliers': int(np.sum(points & ~env)),
            'fill_fraction': float(np.sum(points & env) / max(env.sum(), 1)),
            'bbox': (int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())),
            'centroid': (float(erows.mean()), float(ecols.mean())), 'extent': extent}


def main_cluster_ids(result: ClusterResult, min_size: int = 1, max_clusters: Optional[int] = None) -> List[int]:
    """ids of clusters with size >= min_size (largest first), at most max_clusters"""
    ids = [c.cluster_id for c in result.clusters if c.size >= min_size]
    return ids[:max_clusters] if max_clusters is not None else ids


def cluster_pairs_table(result: ClusterResult, analyzer: UBMatrixAnalyzer,
                        cluster_ids: Optional[List[int]] = None, envelopes: Optional[Dict[int, Dict]] = None,
                        method: str = 'closing', radius: float = 2.0, verbose: bool = True,
                        low_angle: float = 5.0, high_angle: float = 15.0) -> List[Dict[str, Any]]:
    """misorientation between clusters whose envelopes touch or overlap

    misorientation computed between cluster mean orientations (symmetry-aware)
    'overlap': nb of map points in both envelopes (grains superimposed along the beam, or sharing
    the same region), 'contact': nb of points of envelope A adjacent to envelope B
    boundary type: 'low-angle' (< low_angle), 'medium' , 'high-angle' (> high_angle),
                   'Sigma3 twin' (cubic only: 60° about <111> within Brandon criterion 8.66°)
    'axis': misorientation axis in crystal frame of A
    """
    from scipy import ndimage as ndi
    if cluster_ids is None:
        cluster_ids = main_cluster_ids(result, max_clusters=10)
    if envelopes is None:
        envelopes = {cid: cluster_envelope(result, analyzer, cid, method, radius) for cid in cluster_ids}
    stats = result.get_cluster_stats_dict(analyzer)
    symops = get_symmetry_operators(result.symmetry)
    is_cubic = isinstance(result.symmetry, str) and result.symmetry == 'cubic'
    struct = np.ones((3, 3), dtype=bool)
    rows = []
    for a_pos, ca in enumerate(cluster_ids):
        ma = envelopes[ca]['mask']
        grown = ndi.binary_dilation(ma, structure=struct)
        for cb in cluster_ids[a_pos + 1:]:
            mb = envelopes[cb]['mask']
            overlap = int(np.sum(ma & mb))
            contact = int(np.sum(grown & mb & ~ma))
            if overlap == 0 and contact == 0:
                continue
            rot_a, rot_b = stats[ca].mean_orientation, stats[cb].mean_orientation
            angle, axis = misorientation_axis(rot_a, rot_b, symops)
            angle, axis = float(angle[0]), axis[0]
            kind = 'low-angle' if angle < low_angle else ('high-angle' if angle > high_angle else 'medium')
            dev3 = float(sigma3_deviation(rot_a, rot_b)[0]) if is_cubic else np.nan
            if dev3 < BRANDON_SIGMA3:
                kind = 'Sigma3 twin'
            rows.append({'cluster_a': ca, 'cluster_b': cb, 'misorientation': angle, 'axis': axis,
                         'type': kind, 'sigma3_deviation': dev3, 'overlap': overlap, 'contact': contact})
    rows.sort(key=lambda r: r['misorientation'])
    if verbose:
        print(f"{'A':>4} {'B':>4} {'misori(deg)':>11} {'axis (crystal A)':>22} {'type':>12} "
              f"{'dev.Σ3':>7} {'overlap':>8} {'contact':>8}")
        for r in rows:
            ax_txt = '[' + ' '.join(f"{v:6.3f}" for v in r['axis']) + ']'
            print(f"{r['cluster_a']:4d} {r['cluster_b']:4d} {r['misorientation']:11.2f} {ax_txt:>22} "
                  f"{r['type']:>12} {r['sigma3_deviation']:7.2f} {r['overlap']:8d} {r['contact']:8d}")
    return rows


# =============================================================================
# VISUALISATION
# =============================================================================

def _cluster_colormap(nb: int) -> ListedColormap:
    """categorical colormap with nb distinct colors (cycling on tab20, tab20b, tab20c)"""
    base = np.vstack([plt.get_cmap(name).colors for name in ('tab20', 'tab20b', 'tab20c')])
    return ListedColormap(base[np.arange(max(nb, 1)) % len(base)])


def _boundary_segments(labels: np.ndarray) -> np.ndarray:
    """line segments (in imshow pixel coordinates) between points of different labels"""
    segs = []
    diff_h = labels[:, 1:] != labels[:, :-1]  # vertical edges between (r, c) and (r, c+1)
    for r, c in zip(*np.nonzero(diff_h)):
        segs.append([(c + 0.5, r - 0.5), (c + 0.5, r + 0.5)])
    diff_v = labels[1:, :] != labels[:-1, :]  # horizontal edges between (r, c) and (r+1, c)
    for r, c in zip(*np.nonzero(diff_v)):
        segs.append([(c - 0.5, r + 0.5), (c + 0.5, r + 0.5)])
    return np.array(segs).reshape(-1, 2, 2)


def _format_coord(labels_map: np.ndarray, value_map: Optional[np.ndarray] = None, value_name='value',
                  transpose: bool = False):
    """hover text for maps drawn with imshow (any origin): labels_map and value_map in map orientation
    (rows = slow axis, cols = fast axis); transpose=True if the maps were drawn transposed (x = row)"""
    nrows, ncols = labels_map.shape

    def fmt(x, y):
        # pixel (i, j) of the displayed array covers data coordinates [j-0.5, j+0.5[ x [i-0.5, i+0.5[
        j, i = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
        row, col = (j, i) if transpose else (i, j)
        if 0 <= col < ncols and 0 <= row < nrows:
            txt = f"col={col}, row={row}, image={row * ncols + col}, cluster={labels_map[row, col]}"
            if value_map is not None:
                txt += f", {value_name}={value_map[row, col]:.5g}"
            return txt
        return f"x={x:.1f}, y={y:.1f}"
    return fmt


class _MapView:
    """drawing of 2D maps (rows = slow axis, cols = fast axis) with a given imshow origin, optionally
    transposed (x = row, y = col). All map arrays and positions go through it, so that images, markers,
    envelopes, zoom and hover text stay consistent"""

    def __init__(self, origin: str = 'lower', transpose: bool = False):
        if origin not in ('lower', 'upper'):
            raise ValueError("origin must be 'lower' or 'upper'")
        self.origin, self.transpose = origin, transpose

    def arr(self, a: np.ndarray) -> np.ndarray:
        return a.T if self.transpose else a

    def xy(self, row, col):
        """data coordinates (x, y) of map point (row, col)"""
        return (row, col) if self.transpose else (col, row)

    def imshow(self, ax, a: np.ndarray, **kwargs):
        im = ax.imshow(self.arr(a), origin=self.origin, interpolation='nearest', **kwargs)
        # hover text comes only from format_coord() (which includes the value): matplotlib's own
        # cursor value would duplicate it, and is wrong in the 1-pixel band left of / below the map
        im.set_mouseover(False)
        return im

    def envelope(self, ax, mask: np.ndarray, color, **kwargs):
        _draw_envelope(ax, self.arr(mask), color, **kwargs)

    def zoom(self, ax, rmin, rmax, cmin, cmax, margin=1.5):
        bbox = (cmin, cmax, rmin, rmax) if self.transpose else (rmin, rmax, cmin, cmax)
        _zoom_on(ax, bbox, self.origin, margin)

    def labels(self, ax):
        fast, slow = 'fast axis (col)', 'slow axis (row)'
        ax.set_xlabel(slow if self.transpose else fast)
        ax.set_ylabel(fast if self.transpose else slow)

    def format_coord(self, labels_map, value_map=None, value_name='value'):
        return _format_coord(labels_map, value_map, value_name, self.transpose)


def _setup_axes(ax, title, xlabel='fast axis (col)', ylabel='slow axis (row)'):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_cluster_map(result: ClusterResult, analyzer: UBMatrixAnalyzer, mapdimension=None,
                     min_size: int = 1, show_ids: bool = True, show_boundaries: bool = True,
                     ax=None, figsize=(8, 7), origin='lower', output_file: Optional[str] = None,
                     transpose: bool = False, xlabel: Optional[str] = None, ylabel: Optional[str] = None):
    """map of all clusters (one color per cluster, ids written at cluster barycenters)

    at map points with several grains, the grain with most indexed spots is shown
    transpose: swap axes (x = row, slow axis; y = col, fast axis), e.g. to have yech along the vertical axis
    (generaltools.map_transposed()). xlabel, ylabel: axes labels (default: fast/slow axis)
    """
    labels = cluster_label_map(result, analyzer, mapdimension, min_size)
    nb = len(result.clusters)
    view = _MapView(origin, transpose)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    masked = np.ma.masked_less(labels, 0)
    cmap = _cluster_colormap(nb)
    cmap.set_bad('white')
    view.imshow(ax, masked, cmap=cmap, vmin=-0.5, vmax=nb - 0.5)
    if show_boundaries:
        ax.add_collection(LineCollection(_boundary_segments(view.arr(labels)), colors='k', linewidths=0.8))
    if show_ids:
        for cid in np.unique(labels[labels >= 0]):
            rr, cc = np.nonzero(labels == cid)
            k = np.argmin((rr - rr.mean()) ** 2 + (cc - cc.mean()) ** 2)  # point inside the cluster
            ax.text(*view.xy(rr[k], cc[k]), str(cid), ha='center', va='center', fontsize=7, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.6, lw=0))
    shown = len(np.unique(labels[labels >= 0]))
    ax.set_title(f"Clusters (threshold {result.threshold}°, {shown} shown, size >= {min_size})")
    view.labels(ax)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.format_coord = view.format_coord(labels)
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
    return fig, ax


def _cluster_color(result: ClusterResult, cluster_id: int):
    """same color as in plot_cluster_map()"""
    return _cluster_colormap(len(result.clusters)).colors[cluster_id]


def _draw_envelope(ax, mask: np.ndarray, color, linewidth=2.0, fill_alpha=0.0, linestyle='-'):
    """outline (and optional filling) of a boolean map region, in imshow pixel coordinates"""
    nrows, ncols = mask.shape
    padded = np.pad(mask.astype(float), 1)
    xx, yy = np.arange(-1, ncols + 1), np.arange(-1, nrows + 1)
    if fill_alpha > 0:
        ax.contourf(xx, yy, padded, levels=[0.5, 1.5], colors=[color], alpha=fill_alpha)
    ax.contour(xx, yy, padded, levels=[0.5], colors=[color], linewidths=linewidth, linestyles=linestyle)


def _zoom_on(ax, bbox, origin='lower', margin=1.5):
    rmin, rmax, cmin, cmax = bbox
    ax.set_xlim(cmin - margin, cmax + margin)
    ylim = (rmin - margin, rmax + margin)
    ax.set_ylim(ylim if origin == 'lower' else ylim[::-1])


def _gaussian(x, amplitude, mu, sigma):
    return amplitude * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def value_statistics(values: np.ndarray, fit: bool = True, bins=40,
                     percentile_range=(0.5, 99.5)) -> Dict[str, Any]:
    """statistics of the finite values: 'n', 'mean', 'std', 'median', 'min', 'max', and the histogram
    ('counts', 'edges', over percentile_range to exclude outliers, 'nb_out' values outside) with a
    gaussian fit ('fit': [amplitude, mu, sigma] or None if it fails or makes no sense)"""
    from scipy.optimize import curve_fit
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    st = {'n': len(values), 'fit': None, 'counts': None, 'edges': None, 'nb_out': 0}
    if not len(values):
        return st
    std = float(np.std(values))
    if std <= 1e-12 * max(abs(float(np.mean(values))), 1.0):  # constant values (rounding noise)
        std = 0.0
    st.update(mean=float(np.mean(values)), std=std, median=float(np.median(values)),
              min=float(np.min(values)), max=float(np.max(values)))
    low, high = np.percentile(values, percentile_range) if percentile_range else (st['min'], st['max'])
    if high <= low:  # constant values
        low, high = low - 0.5, high + 0.5
    if np.allclose(values, np.round(values)) and high - low < 200:  # integer values: one bin per integer
        edges = np.arange(np.floor(low) - 0.5, np.ceil(high) + 1.0)
    else:
        edges = np.linspace(low, high, bins + 1)
    counts, edges = np.histogram(values, bins=edges)
    st.update(counts=counts, edges=edges, nb_out=int(len(values) - counts.sum()))
    if fit and len(values) >= 5 and st['std'] > 0 and np.count_nonzero(counts) >= 3:
        centers = 0.5 * (edges[:-1] + edges[1:])
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # OptimizeWarning when covariance cannot be estimated
                popt, _ = curve_fit(_gaussian, centers, counts, p0=(counts.max(), st['median'], st['std']),
                                    maxfev=5000)
            popt[2] = abs(popt[2])
            if low - (high - low) <= popt[1] <= high + (high - low):  # reject diverging fits
                st['fit'] = popt
        except (RuntimeError, ValueError):
            pass
    return st


def plot_value_histogram(ax, values: np.ndarray, label: str = 'value', limits=None, fit: bool = True,
                         bins=40, percentile_range=(0.5, 99.5), color='tab:blue') -> Dict[str, Any]:
    """histogram of values with gaussian fit (red) and statistics in the title
    limits: (vmin, vmax) color limits of the corresponding map, drawn as dashed lines
    return value_statistics() dict"""
    st = value_statistics(values, fit, bins, percentile_range)
    if st['n'] == 0:
        ax.set_title('no value', fontsize=9)
        return st
    ax.hist(st['edges'][:-1], bins=st['edges'], weights=st['counts'], color=color, alpha=0.75)
    if st['fit'] is not None:
        xx = np.linspace(st['edges'][0], st['edges'][-1], 300)
        ax.plot(xx, _gaussian(xx, *st['fit']), 'r-', lw=1.5, label='gaussian fit')
    if limits is not None:
        for lim in limits:
            if lim is not None and st['edges'][0] <= lim <= st['edges'][-1]:
                ax.axvline(lim, color='k', ls='--', lw=0.8)
    ax.set_xlabel(label)
    ax.set_ylabel('counts')
    title = (f"N={st['n']}  mean={st['mean']:.4g}  std={st['std']:.3g}\n"
             f"median={st['median']:.4g}  min={st['min']:.4g}  max={st['max']:.4g}")
    if st['fit'] is not None:
        title += f"\ngaussian fit: μ={st['fit'][1]:.4g}  σ={st['fit'][2]:.3g}"
    if st['nb_out']:
        title += f"\n({st['nb_out']} values outside p{percentile_range[0]}-p{percentile_range[1]} not shown)"
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=8)
    return st


def plot_cluster_grod(result: ClusterResult, analyzer: UBMatrixAnalyzer, cluster_id: int,
                      reference: Union[None, str, int, np.ndarray] = None, kind: str = 'grod',
                      mapdimension=None, ax=None, figsize=(7, 6), vmax: Optional[float] = None,
                      cmap=None, origin='lower', zoom: bool = False,
                      show_envelope: bool = False, envelope_method: str = 'closing',
                      envelope_radius: float = 2.0, kam_max_angle: float = 2.0,
                      kam_connectivity: Union[str, int] = '8', verbose: bool = True,
                      output_file: Optional[str] = None, transpose: bool = False, hist_ax=None):
    """map of a single cluster: GROD (misorientation to a reference) or KAM, only this cluster

    all points of the cluster are shown, also where it is not the primary grain of the point.
    kind: 'grod' or 'kam'
    reference (grod): None: point closest to barycenter (red star), 'mean': mean orientation,
                      int: image index of the cluster (red star), (3,3) array: any UB matrix
    show_envelope: outline of the grain region (see cluster_envelope())
    other measured points in light gray. zoom=True restricts the view to the cluster bounding box
    origin: 'lower' or 'upper' (imshow); transpose: swap axes (x = row, slow axis; y = col, fast axis)
    hover text always gives the right (col, row, image index, value)
    hist_ax: axes where to draw the histogram of the values with statistics and gaussian fit
    """
    view = _MapView(origin, transpose)
    nrows, ncols = analyzer.check_mapdimension(mapdimension if mapdimension is not None else result.mapdimension)
    stats = result.get_cluster_stats_dict(analyzer, (nrows, ncols)).get(cluster_id)
    if stats is None:
        print(f"Cluster {cluster_id} not found")
        return None, None
    if kind == 'grod':
        valmap = compute_grod_map(result, analyzer, (nrows, ncols), cluster_id=cluster_id, reference=reference)
        label, cmap = 'misorientation to reference (deg)', cmap or 'viridis'
    elif kind == 'kam':
        valmap = compute_kam(analyzer, result, (nrows, ncols), connectivity=kam_connectivity,
                             max_angle=kam_max_angle, cluster_id=cluster_id)
        label, cmap = 'KAM (deg)', cmap or 'magma'
    else:
        raise ValueError("kind must be 'grod' or 'kam'")
    points = cluster_mask(result, analyzer, cluster_id, (nrows, ncols))
    measured = np.zeros(nrows * ncols, dtype=bool)
    measured[analyzer.image_indices] = True

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    view.imshow(ax, np.where(measured.reshape(nrows, ncols), 1.0, np.nan), cmap=ListedColormap(['0.85']))
    finite = np.isfinite(valmap)
    if kind == 'kam':  # cluster points without valid neighbour
        view.imshow(ax, np.where(points & ~finite, 1.0, np.nan), cmap=ListedColormap(['0.6']))
    top = vmax if vmax is not None else (max(np.nanmax(valmap), 0.1) if finite.any() else 1.0)
    im = view.imshow(ax, valmap, cmap=cmap, vmin=0, vmax=top)
    fig.colorbar(im, ax=ax, label=label, shrink=0.8)
    if hist_ax is not None:
        plot_value_histogram(hist_ax, valmap[finite], label, limits=(0, top))

    env = None
    if show_envelope:
        env = cluster_envelope(result, analyzer, cluster_id, envelope_method, envelope_radius, (nrows, ncols))
        view.envelope(ax, env['mask'], 'red', linewidth=1.5)

    if kind == 'grod':
        if reference is None:
            ref_txt, ref_img = f"ref img {stats.reference_image_index}", stats.reference_image_index
        elif np.ndim(reference) == 0 and not isinstance(reference, str):
            ref_txt, ref_img = f"ref img {int(reference)}", int(reference)
        else:
            ref_txt, ref_img = ('ref mean orientation' if isinstance(reference, str) else 'ref UB given'), None
        if ref_img is not None:
            ax.plot(*view.xy(*divmod(ref_img, ncols)), marker='*', color='red', markersize=12,
                    markeredgecolor='k')
        title = f"Cluster {cluster_id} GROD ({ref_txt}), {int(points.sum())} pts\n"
    else:
        title = f"Cluster {cluster_id} KAM (max angle {kam_max_angle}°), {int(points.sum())} pts\n"
    title += (f"mean {np.nanmean(valmap):.2f}°, max {np.nanmax(valmap):.2f}°" if finite.any() else "no value")
    if env is not None:
        title += f", fill {100 * env['fill_fraction']:.0f}%"
    if zoom:
        view.zoom(ax, stats.min_position[0], stats.max_position[0], stats.min_position[1], stats.max_position[1])
    ax.set_title(title)
    view.labels(ax)
    labels = np.where(points, cluster_id, -1)
    ax.format_coord = view.format_coord(labels, valmap, 'GROD(deg)' if kind == 'grod' else 'KAM(deg)')
    if verbose:
        print(stats)
        print('reference UB matrix:', np.round(stats.reference_matrix, 6).tolist())
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_cluster_property(result: ClusterResult, analyzer: UBMatrixAnalyzer, cluster_id: int,
                          prop: str = 'grod', aligned: bool = True, mapdimension=None, ax=None,
                          figsize=(7, 6), vmin: Optional[float] = None, vmax: Optional[float] = None,
                          percentiles=(2, 98), symmetric: bool = False, cmap=None,
                          strain_multiplier: float = 1e4, origin='lower', zoom: bool = False,
                          show_envelope: bool = False, envelope_radius: float = 2.0,
                          sampletilt: float = 40.0, verbose: bool = True, output_file: Optional[str] = None,
                          transpose: bool = False, hist_ax=None):
    """map of a property of a single cluster (see cluster_property_map() and PROPERTY_GROUPS)

    'grod' and 'kam': see plot_cluster_grod()
    color limits: vmin, vmax, else percentiles of the cluster values; symmetric: limits centered on 0
    strain displayed multiplied by strain_multiplier
    origin: 'lower' or 'upper' (imshow); transpose: swap axes (x = row, slow axis; y = col, fast axis)
    hover text always gives the right (col, row, image index, value)
    hist_ax: axes where to draw the histogram of the values with statistics and gaussian fit
    """
    if prop in ('grod', 'kam'):
        return plot_cluster_grod(result, analyzer, cluster_id, kind=prop, mapdimension=mapdimension, ax=ax,
                                 figsize=figsize, vmax=vmax, cmap=cmap, origin=origin, zoom=zoom,
                                 show_envelope=show_envelope, envelope_radius=envelope_radius,
                                 verbose=verbose, output_file=output_file, transpose=transpose,
                                 hist_ax=hist_ax)
    view = _MapView(origin, transpose)
    nrows, ncols = analyzer.check_mapdimension(mapdimension if mapdimension is not None else result.mapdimension)
    stats = result.get_cluster_stats_dict(analyzer, (nrows, ncols)).get(cluster_id)
    if stats is None:
        print(f"Cluster {cluster_id} not found")
        return None, None
    valmap = cluster_property_map(result, analyzer, cluster_id, prop, aligned, (nrows, ncols), sampletilt)
    if is_strain_property(prop):
        valmap = valmap * strain_multiplier
    if cmap is None:
        cmap = 'bwr' if is_strain_property(prop) else ('magma' if prop in ('nb_indexed', 'pixdev') else 'viridis')
    finite = valmap[np.isfinite(valmap)]
    if finite.size:
        low, high = np.percentile(finite, percentiles)
        if symmetric:
            high = max(abs(low), abs(high))
            low = -high
    else:
        low, high = 0., 1.
    low = low if vmin is None else vmin
    high = high if vmax is None else vmax
    if high <= low:
        high = low + 1e-6

    points = cluster_mask(result, analyzer, cluster_id, (nrows, ncols))
    measured = np.zeros(nrows * ncols, dtype=bool)
    measured[analyzer.image_indices] = True
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    view.imshow(ax, np.where(measured.reshape(nrows, ncols), 1.0, np.nan), cmap=ListedColormap(['0.85']))
    # cluster points without value
    view.imshow(ax, np.where(points & ~np.isfinite(valmap), 1.0, np.nan), cmap=ListedColormap(['0.55']))
    cm = plt.get_cmap(cmap).with_extremes(over='yellow', under='cyan')
    im = view.imshow(ax, valmap, cmap=cm, vmin=low, vmax=high)
    fig.colorbar(im, ax=ax, label=property_label(prop, strain_multiplier), shrink=0.8, extend='both')
    if hist_ax is not None:
        plot_value_histogram(hist_ax, finite, property_label(prop, strain_multiplier), limits=(low, high))
    ax.plot(*view.xy(*divmod(stats.reference_image_index, ncols)), marker='*', color='red', markersize=12,
            markeredgecolor='k')
    if show_envelope:
        env = cluster_envelope(result, analyzer, cluster_id, 'closing', envelope_radius, (nrows, ncols))
        view.envelope(ax, env['mask'], 'k', linewidth=1.5)
    if zoom:
        view.zoom(ax, stats.min_position[0], stats.max_position[0], stats.min_position[1], stats.max_position[1])
    frame_txt = ''
    if prop in CRYSTAL_FRAME_PROPERTIES:
        frame_txt = ' (axes of cluster ref.)' if aligned else ' (axes of written UB)'
    title = f"Cluster {cluster_id}: {property_label(prop, strain_multiplier)}{frame_txt}\n"
    title += (f"{finite.size} pts, median {np.median(finite):.4g}, "
              f"p{percentiles[0]}-p{percentiles[1]} [{np.percentile(finite, percentiles[0]):.4g}, "
              f"{np.percentile(finite, percentiles[1]):.4g}]" if finite.size else 'no value')
    ax.set_title(title)
    view.labels(ax)
    ax.title.set_fontsize(10)
    ax.format_coord = view.format_coord(np.where(points, cluster_id, -1), valmap, prop)
    if verbose:
        print(stats)
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
    return fig, ax


def _property_options() -> List[Tuple[str, str]]:
    """(label, property) options for the interactive browser"""
    options = []
    for group, props in PROPERTY_GROUPS.items():
        for prop in props:
            options.append((f"{group}: {prop}", prop))
    options.insert(1, ('orientation: kam', 'kam'))
    return options


def plot_single_cluster(result: ClusterResult, analyzer: UBMatrixAnalyzer, cluster_id: int,
                        mapdimension=None, ax=None, figsize=(7, 6), vmax: Optional[float] = None,
                        cmap='viridis', origin='lower', zoom: bool = False, verbose: bool = True,
                        output_file: Optional[str] = None, reference=None, show_envelope: bool = False):
    """map of one cluster colored by misorientation (deg) to its reference (GROD of this cluster)"""
    return plot_cluster_grod(result, analyzer, cluster_id, reference=reference, kind='grod',
                             mapdimension=mapdimension, ax=ax, figsize=figsize, vmax=vmax, cmap=cmap,
                             origin=origin, zoom=zoom, show_envelope=show_envelope, verbose=verbose,
                             output_file=output_file)


def plot_cluster_grod_kam(result: ClusterResult, analyzer: UBMatrixAnalyzer, cluster_id: int,
                          reference: Union[None, str, int, np.ndarray] = None, zoom: bool = True,
                          show_envelope: bool = True, envelope_radius: float = 2.0,
                          kam_max_angle: float = 2.0, kam_connectivity: Union[str, int] = '8',
                          vmax_grod: Optional[float] = None, vmax_kam: Optional[float] = None,
                          figsize=(13, 5.5), verbose: bool = True, output_file: Optional[str] = None):
    """GROD and KAM of a single cluster side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    plot_cluster_grod(result, analyzer, cluster_id, reference=reference, kind='grod', ax=ax1,
                      vmax=vmax_grod, zoom=zoom, show_envelope=show_envelope,
                      envelope_radius=envelope_radius, verbose=verbose)
    plot_cluster_grod(result, analyzer, cluster_id, kind='kam', ax=ax2, vmax=vmax_kam, zoom=zoom,
                      show_envelope=show_envelope, envelope_radius=envelope_radius,
                      kam_max_angle=kam_max_angle, kam_connectivity=kam_connectivity, verbose=False)
    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
    return fig


def format_matrix_list(matrix: np.ndarray, decimals: int = 9) -> str:
    """matrix as a python nested list string, e.g. '[[0.858773329, -0.486781455, ...], [...], [...]]',
    ready to copy-paste into LaueTools GUIs or notebooks"""
    rows = [', '.join(f"{v:.{decimals}f}" for v in line) for line in np.asarray(matrix, dtype=float)]
    return '[' + ', '.join(f'[{r}]' for r in rows) + ']'


def _cluster_stats_html(result: ClusterResult, analyzer: UBMatrixAnalyzer, cluster_id: int) -> str:
    """short html summary of a cluster for the interactive browsers"""
    s = result.get_cluster_stats_dict(analyzer)[cluster_id]
    row, col = s.mean_position
    return (f"<b>Cluster {s.cluster_id}</b>: {s.nb_pixels} pts ({s.size} matrices)<br>"
            f"mean (col, row) = ({col:.1f}, {row:.1f})<br>"
            f"std {s.std_orientation:.2f}°, max dev {s.max_angular_deviation:.2f}°<br>"
            f"grains {sorted(set(s.grain_indices))}<br>"
            f"ref: image {s.reference_image_index}, "
            f"{analyzer.matrices[s.reference_matrix_index].filename}")


def _offscreen_figure(figsize) -> plt.Figure:
    """figure NOT registered in pyplot: never shown by plt.show(), no live ipympl canvas, freed with
    its last reference (light for JupyterLab). Render it with _figure_png()"""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    return fig


def _live_figure(figsize) -> Optional[plt.Figure]:
    """interactive ipympl figure (zoom, hover values) NOT registered in pyplot: plt.show() does not
    display it a second time. None if ipympl is not installed. Close it with fig.canvas.close()"""
    try:
        from ipympl.backend_nbagg import Canvas, FigureManager
    except ImportError:
        return None
    from matplotlib.figure import Figure
    fig = Figure(figsize=figsize)
    canvas = Canvas(fig)
    FigureManager(canvas, 0)
    return fig


def _figure_png(fig, dpi: int = 90) -> bytes:
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi)
    return buf.getvalue()


def _cluster_browser(result: ClusterResult, analyzer: UBMatrixAnalyzer, draw, min_size: int = 1,
                     figsize=(7, 6), extra_widgets: Optional[Dict[str, Any]] = None, rows: int = 12,
                     live: bool = False, dpi: int = 90):
    """Jupyter cluster browser: list box of clusters (arrow keys up/down redraw immediately),
    previous/next buttons, optional extra widgets, cluster summary, reference UB matrix (copyable
    list) and the figure

    draw(fig, cluster_id, **values of extra_widgets) draws in the (cleared) figure
    'live plot' checkbox (initial value: live):
    - unchecked (default): the figure is rendered as a PNG image (light and fast in JupyterLab)
    - checked: interactive matplotlib canvas (ipympl): zoom, hover coordinates and values. The canvas
      is not registered in pyplot (never duplicated by plt.show()); it is created at first use, then
      only hidden / shown by the checkbox
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print("Install ipywidgets: pip install ipywidgets")
        return
    valid = [c for c in result.clusters if c.size >= min_size]
    if not valid:
        print(f"No clusters with size >= {min_size}")
        return
    options = [(f"Cluster {c.cluster_id} ({c.nb_pixels} pts)", c.cluster_id) for c in valid]
    ids = [cid for _, cid in options]
    extras = extra_widgets or {}
    width = widgets.Layout(width='210px')

    selector = widgets.Select(options=options, value=ids[0], rows=min(rows, len(options)), layout=width)
    prev_btn = widgets.Button(description='◀ prev', layout=widgets.Layout(width='102px'))
    next_btn = widgets.Button(description='next ▶', layout=widgets.Layout(width='102px'))
    hint = widgets.HTML("<small>click in the list, then use ↑ ↓ keys</small>")
    live_w = widgets.Checkbox(value=False, description='live plot (zoom, values)', indent=False, layout=width)
    info = widgets.HTML(layout=width)
    ub_label = widgets.HTML("<small>reference UB matrix (click, Ctrl+A, Ctrl+C):</small>")
    ub_text = widgets.Textarea(layout=widgets.Layout(width='210px', height='95px'))
    image = widgets.Image(format='png')
    figure_box = widgets.Box([image])
    # static figure (PNG image) and live figure (ipympl canvas, created at first use) both stay in
    # figure_box: the checkbox only hides one and shows the other (views are never closed or recreated,
    # which would leave a frozen copy of the canvas in the notebook)
    state = {'static_fig': _offscreen_figure(figsize), 'live_fig': None, 'live': False}

    def update(*_):
        fig = state['live_fig'] if state['live'] else state['static_fig']
        cluster_id = selector.value
        try:
            fig.clf()
            draw(fig, cluster_id, **{name: w.value for name, w in extras.items()})
            info.value = _cluster_stats_html(result, analyzer, cluster_id)
            ub_text.value = format_matrix_list(result.get_cluster_stats_dict(analyzer)[cluster_id].reference_matrix)
        except Exception as err:  # errors in widget callbacks are otherwise silent
            info.value = f"<span style='color:red'>Error: {err}</span>"
        if state['live']:
            fig.canvas.draw_idle()
        else:
            image.value = _figure_png(fig, dpi)

    def set_live(change):
        if change['new']:
            if state['live_fig'] is None:
                fig = _live_figure(figsize)
                if fig is None:
                    live_w.value = False
                    info.value += ("<br><span style='color:red'>live plot needs ipympl "
                                   "(pip install ipympl, %matplotlib widget)</span>")
                    return
                state['live_fig'] = fig
                figure_box.children = [image, fig.canvas]
            state['live'] = True
            image.layout.display = 'none'
            state['live_fig'].canvas.layout.display = None
        else:
            if not state['live']:
                return
            state['live'] = False
            state['live_fig'].canvas.layout.display = 'none'
            image.layout.display = None
        update()

    def step(delta):
        selector.value = ids[(ids.index(selector.value) + delta) % len(ids)]

    prev_btn.on_click(lambda _: step(-1))
    next_btn.on_click(lambda _: step(1))
    selector.observe(update, names='value')
    for w in extras.values():
        w.observe(update, names='value')
    live_w.observe(set_live, names='value')

    controls = widgets.VBox([hint, selector, widgets.HBox([prev_btn, next_btn]), *extras.values(),
                             live_w, info, ub_label, ub_text])
    display(widgets.HBox([controls, figure_box]))
    update()
    if live:
        live_w.value = True


def cluster_grod_interactive(result: ClusterResult, analyzer: UBMatrixAnalyzer, min_size: int = 8,
                             zoom: bool = True, show_envelope: bool = True, figsize=(12, 5),
                             kam_max_angle: float = 2.0, live: bool = False):
    """Jupyter browser: GROD and KAM of one cluster, reference 'barycenter point' or 'mean'

    click in the cluster list then browse with arrow keys: the maps are redrawn immediately
    """
    try:
        import ipywidgets as widgets
    except ImportError:
        print("Install ipywidgets: pip install ipywidgets")
        return
    ref_w = widgets.RadioButtons(options=[('barycenter point', 'barycenter'), ('mean orientation', 'mean')],
                                 value='barycenter', description='Reference:',
                                 layout=widgets.Layout(width='210px'),
                                 style={'description_width': 'initial'})

    def draw(fig, cluster_id, reference):
        ax1, ax2 = fig.subplots(1, 2)
        ref = None if reference == 'barycenter' else reference
        plot_cluster_grod(result, analyzer, cluster_id, reference=ref, kind='grod', ax=ax1, zoom=zoom,
                          show_envelope=show_envelope, verbose=False)
        plot_cluster_grod(result, analyzer, cluster_id, kind='kam', ax=ax2, zoom=zoom,
                          show_envelope=show_envelope, kam_max_angle=kam_max_angle, verbose=False)
        fig.tight_layout()

    _cluster_browser(result, analyzer, draw, min_size=min_size, figsize=figsize, live=live,
                     extra_widgets={'reference': ref_w})


def plot_cluster_envelopes(result: ClusterResult, analyzer: UBMatrixAnalyzer,
                           cluster_ids: Optional[List[int]] = None, min_size: int = 100,
                           max_clusters: int = 10, method: str = 'closing', radius: float = 2.0,
                           show_points: bool = True, fill_alpha: float = 0.15,
                           background: Optional[np.ndarray] = None, background_cmap='Greys',
                           ax=None, figsize=(9, 8), origin='lower', legend: bool = True,
                           output_file: Optional[str] = None):
    """main clusters superimposed: envelope of each grain (outline + light filling) and its points

    sparse clusters: the envelope (see cluster_envelope(), method 'closing' with radius in map steps,
    'hull' for convex extreme limits) shows the region occupied by the grain.
    Overlapping envelopes: grains superimposed along the beam (several grains per image) or
    interpenetrating grains.
    cluster_ids: clusters to show (default: the max_clusters largest with size >= min_size)
    background: optional 2D map shown in gray (e.g. KAM)
    return fig, ax, dict cluster_id -> envelope
    """
    nrows, ncols = analyzer.check_mapdimension(result.mapdimension)
    if cluster_ids is None:
        cluster_ids = main_cluster_ids(result, min_size, max_clusters)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    if background is not None:
        cm = plt.get_cmap(background_cmap).copy()
        cm.set_bad('white')
        ax.imshow(background, cmap=cm, origin=origin, interpolation='nearest', alpha=0.5,
                  vmax=np.nanpercentile(background, 99))
    else:
        measured = np.zeros(nrows * ncols, dtype=bool)
        measured[analyzer.image_indices] = True
        ax.imshow(np.where(measured.reshape(nrows, ncols), 1.0, np.nan), cmap=ListedColormap(['0.93']),
                  origin=origin, interpolation='nearest')

    envelopes = {}
    handles = []
    marker_size = max(4.0, 2500.0 / max(nrows, ncols) ** 1.5)
    for k, cid in enumerate(cluster_ids):
        env = cluster_envelope(result, analyzer, cid, method, radius, (nrows, ncols))
        envelopes[cid] = env
        color = _cluster_color(result, cid)
        _draw_envelope(ax, env['mask'], color, linewidth=2.0, fill_alpha=fill_alpha)
        if show_points:
            prow, pcol = np.nonzero(env['points'])
            # small shift per cluster so that superimposed grains at the same point remain visible
            shift = 0.18 * np.array([np.cos(2.4 * k), np.sin(2.4 * k)])
            ax.scatter(pcol + shift[0], prow + shift[1], s=marker_size, color=color, marker='s',
                       edgecolors='none', alpha=0.9)
        erow, ecol = np.nonzero(env['mask'])
        j = np.argmin((erow - env['centroid'][0]) ** 2 + (ecol - env['centroid'][1]) ** 2)
        ax.text(ecol[j], erow[j], str(cid), ha='center', va='center', fontsize=10, fontweight='bold',
                color='k', bbox=dict(boxstyle='round,pad=0.15', fc=color, alpha=0.85, lw=0))
        handles.append(plt.Line2D([], [], color=color, lw=3,
                                  label=f"{cid}: {env['nb_points']} pts, fill {100 * env['fill_fraction']:.0f}%"))
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim((-0.5, nrows - 0.5) if origin == 'lower' else (nrows - 0.5, -0.5))
    if legend and handles:
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8,
                  title=f"envelope: {method}" + (f" r={radius}" if method == 'closing' else ''))
    _setup_axes(ax, f"Main clusters ({len(cluster_ids)}) and their envelopes")
    ax.format_coord = _format_coord(cluster_label_map(result, analyzer))
    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
    return fig, ax, envelopes


def boundary_segments_misorientation(result: ClusterResult, analyzer: UBMatrixAnalyzer,
                                     mapdimension=None) -> Dict[str, np.ndarray]:
    """boundary segments between neighbouring map points (edges) whose primary grains belong to
    different clusters, with the misorientation (deg) between the two primary matrices

    return dict: 'segments' (n, 2, 2) in imshow coordinates, 'angles' (n,), 'clusters' (n, 2)
    """
    pmap = primary_matrix_map(analyzer, result, mapdimension)
    labels = np.where(pmap >= 0, result.cluster_assignment[np.maximum(pmap, 0)], -1)
    segs, ia, ib = [], [], []
    for (dr, dc) in ((0, 1), (1, 0)):
        a = labels[:labels.shape[0] - dr, :labels.shape[1] - dc]
        b = labels[dr:, dc:]
        rr, cc = np.nonzero((a >= 0) & (b >= 0) & (a != b))
        for r, c in zip(rr, cc):
            if dc:  # vertical edge between (r, c) and (r, c+1)
                segs.append([(c + 0.5, r - 0.5), (c + 0.5, r + 0.5)])
            else:   # horizontal edge between (r, c) and (r+1, c)
                segs.append([(c - 0.5, r + 0.5), (c + 0.5, r + 0.5)])
            ia.append(pmap[r, c])
            ib.append(pmap[r + dr, c + dc])
    ia, ib = np.array(ia, dtype=int), np.array(ib, dtype=int)
    angles = (misorientation(analyzer.rotations[ia], analyzer.rotations[ib],
                             get_symmetry_operators(result.symmetry))[0] if len(ia) else np.zeros(0))
    is_cubic = isinstance(result.symmetry, str) and result.symmetry == 'cubic'
    dev3 = (sigma3_deviation(analyzer.rotations[ia], analyzer.rotations[ib]) if (is_cubic and len(ia))
            else np.full(len(ia), np.nan))
    return {'segments': np.array(segs).reshape(-1, 2, 2), 'angles': angles, 'sigma3_deviation': dev3,
            'clusters': np.column_stack([result.cluster_assignment[ia], result.cluster_assignment[ib]])
            if len(ia) else np.zeros((0, 2), dtype=int)}


def plot_boundary_map(result: ClusterResult, analyzer: UBMatrixAnalyzer, low_angle: float = 5.0,
                      high_angle: float = 15.0, background: Union[str, np.ndarray, None] = 'clusters',
                      ax=None, figsize=(9, 8), origin='lower', linewidth=1.5,
                      output_file: Optional[str] = None):
    """grain boundary map: boundaries between neighbouring points of different clusters, colored by
    misorientation class: low-angle (< low_angle, blue), medium (orange), high-angle (> high_angle, black)
    and, for cubic symmetry, Sigma3 twin boundaries (60° about <111>, Brandon criterion, red)

    background: 'clusters' (light cluster map), None, or any 2D map (e.g. KAM) shown in gray
    NB: at points with several grains, the primary one (most indexed spots) is used, so boundaries
    also appear where the primary grain switches between superimposed grains
    """
    nrows, ncols = analyzer.check_mapdimension(result.mapdimension)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    labels = cluster_label_map(result, analyzer)
    if isinstance(background, str) and background == 'clusters':
        cmap = _cluster_colormap(len(result.clusters))
        ax.imshow(np.ma.masked_less(labels, 0), cmap=cmap, vmin=-0.5, vmax=len(result.clusters) - 0.5,
                  origin=origin, interpolation='nearest', alpha=0.35)
    elif background is not None:
        cm = plt.get_cmap('Greys').copy()
        cm.set_bad('white')
        ax.imshow(background, cmap=cm, origin=origin, interpolation='nearest', alpha=0.6,
                  vmax=np.nanpercentile(background, 99))
    bnd = boundary_segments_misorientation(result, analyzer)
    twin = bnd['sigma3_deviation'] < BRANDON_SIGMA3
    classes = [('low-angle', bnd['angles'] < low_angle, 'tab:blue', f'< {low_angle}°'),
               ('medium', (bnd['angles'] >= low_angle) & (bnd['angles'] <= high_angle) & ~twin, 'tab:orange',
                f'{low_angle}-{high_angle}°'),
               ('high-angle', (bnd['angles'] > high_angle) & ~twin, 'k', f'> {high_angle}°')]
    if isinstance(result.symmetry, str) and result.symmetry == 'cubic':
        classes.append(('Σ3 twin', twin, 'red', '60°<111>'))
    handles = []
    for name, sel, color, rng in classes:
        if np.any(sel):
            ax.add_collection(LineCollection(bnd['segments'][sel], colors=color, linewidths=linewidth))
        handles.append(plt.Line2D([], [], color=color, lw=2, label=f"{name} {rng}: {int(np.sum(sel))}"))
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim((-0.5, nrows - 0.5) if origin == 'lower' else (nrows - 0.5, -0.5))
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8,
              title='boundary segments')
    _setup_axes(ax, 'Cluster boundaries colored by misorientation')
    ax.format_coord = _format_coord(labels)
    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
    return fig, ax, bnd


def plot_clusters_gallery(result: ClusterResult, analyzer: UBMatrixAnalyzer, mapdimension=None,
                          min_size: int = 1, max_clusters: int = 24, nb_cols: int = 4,
                          vmax: Optional[float] = None, zoom: bool = False,
                          panel_size: float = 3.2, origin='lower', show_envelope: bool = False,
                          envelope_radius: float = 2.0, output_file: Optional[str] = None,
                          live: bool = False, dpi: int = 80, static: Optional[bool] = None):
    """small multiples: one map per cluster (largest first), colored by misorientation to reference

    vmax: common color scale (deg) for all panels; None: each panel scaled to its own max deviation
    show_envelope: outline of each grain region (see cluster_envelope())
    live=False (default): the figure is displayed once as a PNG image (light for JupyterLab)
    live=True: interactive canvas (ipympl): zoom, hover coordinates and values
    in both cases the figure is not registered in pyplot (never duplicated by plt.show())
    static: deprecated, static=True is live=False
    """
    if static is not None:
        live = not static
    selected = [c.cluster_id for c in result.clusters if c.size >= min_size][:max_clusters]
    if not selected:
        print(f"No cluster with size >= {min_size}")
        return None
    nb_rows = int(np.ceil(len(selected) / nb_cols))
    figsize = (panel_size * nb_cols, panel_size * nb_rows)
    fig = _live_figure(figsize) if live else None
    if live and fig is None:
        print("live=True needs ipympl (pip install ipympl): static image shown")
    is_live = fig is not None
    if not is_live:
        fig = _offscreen_figure(figsize)
    axes = fig.subplots(nb_rows, nb_cols, squeeze=False)
    for ax, cid in zip(axes.flat, selected):
        plot_cluster_grod(result, analyzer, cid, mapdimension=mapdimension, ax=ax, vmax=vmax, zoom=zoom,
                          origin=origin, show_envelope=show_envelope, envelope_radius=envelope_radius,
                          verbose=False)
        ax.title.set_fontsize(8)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(selected):]:
        ax.axis('off')
    for cax in fig.axes[axes.size:]:  # colorbars
        cax.tick_params(labelsize=7)
        cax.yaxis.label.set_fontsize(7)
    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=200, bbox_inches='tight')
    try:
        from IPython.display import display, Image
        if is_live:
            display(fig.canvas)
            fig.canvas.draw_idle()
        else:
            display(Image(data=_figure_png(fig, dpi)))
    except ImportError:
        pass
    return fig


def visualize_cluster_interactive(result: ClusterResult, analyzer: UBMatrixAnalyzer,
                                  mapdimension=None, min_size: int = 1, figsize=None, zoom=False,
                                  properties: bool = True, prop: str = 'grod', live: bool = False,
                                  origin: str = 'lower', transpose: bool = False, histogram: bool = True):
    """Jupyter browser to show clusters one by one (requires ipywidgets)

    click in the cluster list then browse with arrow keys: the map is redrawn immediately
    properties=True: menu to choose the property shown (GROD, KAM, Euler angles, strain in sample or
    crystal frame, lattice parameters, quality), each value read from the .fit file of the matrix of the
    cluster at each point. Crystal-frame quantities can be aligned to the axes of the cluster reference.
    live=False: maps rendered as PNG images (light, fast); True: interactive ipympl canvas (heavier)
    origin ('lower' / 'upper') and transpose (swap axes, x = row) also set by widgets in the browser;
    hover text in live mode always gives the right (col, row, image index, value)
    histogram: histogram of the displayed values, statistics and gaussian fit next to the map
    (also a checkbox in the browser)
    """
    if figsize is None:
        figsize = (12, 5.5) if properties else (7, 6)
    if not properties:
        def draw(fig, cluster_id):
            ax = fig.add_subplot(111)
            plot_single_cluster(result, analyzer, cluster_id, mapdimension, ax=ax, zoom=zoom, verbose=False)

        _cluster_browser(result, analyzer, draw, min_size=min_size, figsize=figsize, live=live)
        return

    import ipywidgets as widgets
    width = widgets.Layout(width='210px')
    prop_w = widgets.Dropdown(options=_property_options(), value=prop, layout=width)
    aligned_w = widgets.Checkbox(value=True, description='crystal axes of cluster ref.', indent=False, layout=width)
    symmetric_w = widgets.Checkbox(value=False, description='color scale centered on 0', indent=False, layout=width)
    zoom_w = widgets.Checkbox(value=zoom, description='zoom on cluster', indent=False, layout=width)
    envelope_w = widgets.Checkbox(value=False, description='envelope', indent=False, layout=width)
    origin_w = widgets.ToggleButtons(options=['lower', 'upper'], value=origin, description='origin',
                                     style={'button_width': '70px', 'description_width': '40px'},
                                     layout=width)
    transpose_w = widgets.Checkbox(value=transpose, description='swap axes (x = row)', indent=False,
                                   layout=width)
    histogram_w = widgets.Checkbox(value=histogram, description='histogram, stats, gaussian fit', indent=False,
                                   layout=width)

    def draw(fig, cluster_id, prop, aligned, symmetric, zoom, envelope, origin, transpose, histogram):
        if histogram:
            grid = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.35)
            ax, hist_ax = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
        else:
            ax, hist_ax = fig.add_subplot(111), None
        plot_cluster_property(result, analyzer, cluster_id, prop, aligned=aligned, mapdimension=mapdimension,
                              ax=ax, symmetric=symmetric, zoom=zoom, show_envelope=envelope, verbose=False,
                              origin=origin, transpose=transpose, hist_ax=hist_ax)

    _cluster_browser(result, analyzer, draw, min_size=min_size, figsize=figsize, live=live,
                     extra_widgets={'prop': prop_w, 'aligned': aligned_w, 'symmetric': symmetric_w,
                                    'zoom': zoom_w, 'envelope': envelope_w, 'origin': origin_w,
                                    'transpose': transpose_w, 'histogram': histogram_w})


def plot_kam(kam: np.ndarray, result: Optional[ClusterResult] = None,
             analyzer: Optional[UBMatrixAnalyzer] = None, vmax: Optional[float] = None,
             vmin: float = 0.0,
             cmap='magma', show_boundaries: bool = True, ax=None, figsize=(8, 7), origin='lower',
             title='Kernel Average Misorientation (deg)', output_file: Optional[str] = None):
    """plot a KAM (or GROD) map, with cluster boundaries if result and analyzer are given

    vmax: default 99th percentile of the map; vmin: default 0 (any 2D map, e.g. OC.property_map())
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    if vmax is None:
        vmax = np.nanpercentile(kam, 99) if np.any(np.isfinite(kam)) else 1.0
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad('0.85')
    im = ax.imshow(kam, cmap=cm, vmin=vmin, vmax=vmax, origin=origin, interpolation='nearest')
    fig.colorbar(im, ax=ax, shrink=0.8, extend='max' if vmin == 0 else 'both')
    labels = np.full(kam.shape, -1)
    if result is not None and analyzer is not None:
        labels = cluster_label_map(result, analyzer, kam.shape)
        if show_boundaries:
            ax.add_collection(LineCollection(_boundary_segments(labels), colors='cyan', linewidths=0.6))
    _setup_axes(ax, title)
    ax.format_coord = _format_coord(labels, kam, 'value')
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
    return fig, ax
