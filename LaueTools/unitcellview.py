#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Drawings of the unit cell of grains with their orientation (UB matrix of LaueTools) in the sample frame

Drawings in the BM32 laboratory view: incoming beam horizontal pointing to the left, sample surface tilted by 40 deg,
Ys toward the viewer, Zs pointing up right.

Sample frame of LaueTools (CrystalParameters.strain_from_crystal_to_sample_frame2): Zs normal to the sample surface,
Xs in the plane of the incoming beam and Zs (tilted by the sample tilt from the beam), Ys horizontal = Y of LaueTools.
q_LT = P q_sample with P = rotation of -tilt around Y.

Direct basis (a, b, c) in the LaueTools frame from q_LT = UB B0 G*: [a b c] = ((UB B0)^-1)^T (transpose of the
inverse of the reciprocal basis), B0 from the lattice parameters of the material.

Main functions: cell_basis_sample(), cell_edges(), plot_cell(), plot_cell_views(), CellBrowser (one grain at a time,
ipywidgets slider), grains_from_fitfiles().
"""
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    from . import CrystalParameters as CP
    from . import dict_LaueTools as DictLT
    from . import generaltools as GT
except ImportError:
    import CrystalParameters as CP
    import dict_LaueTools as DictLT
    import generaltools as GT


def lattice_parameters(key_material: str, dictmaterials=None) -> List[float]:
    """[a, b, c, alpha, beta, gamma] of a material of the LaueTools dictionary"""
    dictmaterials = dictmaterials or DictLT.dict_Materials
    return [float(v) for v in dictmaterials[key_material][1]]


def is_hexagonal(latticeparams: Sequence[float]) -> bool:
    a, b, c, alpha, beta, gamma = latticeparams
    return bool(np.isclose(a, b) and np.allclose([alpha, beta], 90.) and np.isclose(gamma, 120.))


def cell_basis_sample(UB, latticeparams: Sequence[float], sample_tilt: float = 40.) -> np.ndarray:
    """columns: direct lattice vectors a, b, c (angstrom) of the grain in the sample frame"""
    B0 = CP.calc_B_RR(list(latticeparams))
    recip = np.asarray(UB, dtype=float) @ B0
    direct = np.linalg.inv(recip).T
    # lengths of the lattice parameters (the scale of B0 and of UB does not matter)
    lengths = np.linalg.norm(direct, axis=0)
    direct = direct / lengths * np.asarray(latticeparams[:3], dtype=float)
    P = GT.matRot([0, 1, 0], -sample_tilt)   # q_LT = P q_sample
    return P.T @ direct


def cell_edges(basis: np.ndarray, hexagonal: bool = False) -> List[np.ndarray]:
    """segments (2 x 3 arrays) of the edges of the cell: parallelepiped (a, b, c), or hexagonal prism (a1, a2, c,
    6 sides of length a around the c axis) centered on the origin"""
    a1, a2, c = basis[:, 0], basis[:, 1], basis[:, 2]
    if hexagonal:
        base = [a1, a1 + a2, a2, -a1, -a1 - a2, -a2]
        lower = [v - 0.5 * c for v in base]
        upper = [v + 0.5 * c for v in base]
        edges = []
        for k in range(6):
            edges.append(np.array([lower[k], lower[(k + 1) % 6]]))
            edges.append(np.array([upper[k], upper[(k + 1) % 6]]))
            edges.append(np.array([lower[k], upper[k]]))
        return edges
    center = 0.5 * (a1 + a2 + c)
    corners = {(i, j, k): i * a1 + j * a2 + k * c - center for i in (0, 1) for j in (0, 1) for k in (0, 1)}
    edges = []
    for (i, j, k), p in corners.items():
        for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            q = (i + d[0], j + d[1], k + d[2])
            if q in corners:
                edges.append(np.array([p, corners[q]]))
    return edges


def direction_indices(basis: np.ndarray, vector_sample) -> np.ndarray:
    """crystal direction [u v w] (direct basis of the cell, largest component 1) along a vector of the sample frame"""
    uvw = np.linalg.solve(basis, np.asarray(vector_sample, dtype=float))
    return uvw / np.max(np.abs(uvw))


# BM32 laboratory view: LaueTools frame (x along the incoming beam, z vertical up) -> display (H horizontal to the
# right, V vertical up, D toward the viewer): beam horizontal pointing to the left, Ys toward the viewer, Zs (normal
# of the sample tilted by 40 deg) pointing up right
_LT_TO_DISPLAY = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
DEFAULT_VIEW = (15., -65.)   # elevation, azimuth (deg) of the 3D view


def sample_to_display(vectors, sample_tilt: float = 40.) -> np.ndarray:
    """(H, V, D) laboratory display coordinates of vectors (3 or 3 x n) of the sample frame"""
    return _LT_TO_DISPLAY @ GT.matRot([0, 1, 0], -sample_tilt) @ np.asarray(vectors, dtype=float)


def _mpl(v):
    """display (H, V, D) -> matplotlib 3D axes (x = H, y = -D: into the screen for azimuth -90, z = V)"""
    v = np.asarray(v, dtype=float)
    return np.array([v[0], -v[2], v[1]]) if v.ndim == 1 else np.vstack([v[0], -v[2], v[1]])


def _arrow3d(ax, vector, color, label):
    v = _mpl(vector)
    ax.quiver(0, 0, 0, *v, color=color, arrow_length_ratio=0.12, lw=1.5)
    ax.text(*(1.12 * v), label, color=color, fontsize=9)


def plot_cell(ax, UB, latticeparams: Sequence[float], sample_tilt: float = 40., color='tab:blue',
              axes_labels: bool = True, title: Optional[str] = None, view: Optional[Sequence[float]] = None):
    """3D drawing (matplotlib axes with projection='3d') of the cell in the BM32 laboratory view: incoming beam
    (orange) horizontal pointing to the left, sample surface (grey plane) tilted by sample_tilt, Ys toward the
    viewer, Zs up right; axes of the cell a (red), b (green), c (blue), sample frame (grey)

    :param view: (elevation, azimuth) of the view (default DEFAULT_VIEW)
    """
    basis = cell_basis_sample(UB, latticeparams, sample_tilt)
    disp = sample_to_display(basis, sample_tilt)
    size = 1.6 * np.max(np.linalg.norm(basis, axis=0))
    for edge in cell_edges(disp, is_hexagonal(latticeparams)):
        ax.plot(*_mpl(edge.T), color=color, lw=1.4)
    for k, (name, col) in enumerate((('a', 'tab:red'), ('b', 'tab:green'), ('c', 'tab:blue'))):
        _arrow3d(ax, disp[:, k], col, name)
    if axes_labels:
        frame = sample_to_display(np.eye(3), sample_tilt)
        for k, name in enumerate(('Xs', 'Ys', 'Zs')):
            _arrow3d(ax, frame[:, k] * size * 0.9, '0.4', name)
        _arrow3d(ax, np.array([-1., 0., 0.]) * size * 0.95, 'orange', 'beam')
        # sample surface: plane (Xs, Ys) below the cell
        u, v = np.meshgrid([-1.2 * size, 1.2 * size], [-size, size])
        origin = -0.85 * size * frame[:, 2]
        pts = origin[:, None, None] + frame[:, 0][:, None, None] * u + frame[:, 1][:, None, None] * v
        X, Y, Z = _mpl(pts.reshape(3, -1))
        ax.plot_surface(X.reshape(u.shape), Y.reshape(u.shape), Z.reshape(u.shape), alpha=0.15, color='0.5')
    for setlim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setlim(-size, size)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    ax.set_xlabel('horizontal')
    ax.set_ylabel('toward viewer (Ys)')
    ax.view_init(*(view if view is not None else DEFAULT_VIEW))
    if title:
        ax.set_title(title, fontsize=9)
    return basis


def plot_cell_projections(axes, UB, latticeparams: Sequence[float], sample_tilt: float = 40., color='tab:blue'):
    """projections of the cell on the planes (Xs, Ys) (seen from above the sample), (Xs, Zs) and (Ys, Zs)"""
    basis = cell_basis_sample(UB, latticeparams, sample_tilt)
    edges = cell_edges(basis, is_hexagonal(latticeparams))
    size = 1.6 * np.max(np.linalg.norm(basis, axis=0))
    beam = GT.matRot([0, 1, 0], -sample_tilt).T @ np.array([1., 0., 0.])
    # first view: BM32 laboratory side view (along Ys toward the viewer): beam to the left, surface tilted
    ax = axes[0]
    ax.clear()
    disp = sample_to_display(basis, sample_tilt)
    frame = sample_to_display(np.eye(3), sample_tilt)
    for edge in cell_edges(disp, is_hexagonal(latticeparams)):
        ax.plot(edge[:, 0], edge[:, 1], color=color, lw=1.2)
    for k, col in enumerate(('tab:red', 'tab:green', 'tab:blue')):
        ax.annotate('', xy=(disp[0, k], disp[1, k]), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=col, lw=1.5))
    surface = -0.6 * size * frame[:, 2]   # sample surface (grey line, tilted by sample_tilt) below the cell
    ax.plot([surface[0] - 2. * size * frame[0, 0], surface[0] + 2. * size * frame[0, 0]],
            [surface[1] - 2. * size * frame[1, 0], surface[1] + 2. * size * frame[1, 0]], color='0.5', lw=4,
            alpha=0.5)
    for k, name in ((0, 'Xs'), (2, 'Zs')):
        tip = 0.9 * size * frame[:2, k]
        ax.annotate('', xy=tuple(tip), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='0.4', lw=1))
        ax.text(*(1.08 * tip), name, color='0.4', fontsize=8, ha='center', va='center')
    ax.annotate('', xy=(-0.95 * size, 0.), xytext=(0.95 * size, 0.),
                arrowprops=dict(arrowstyle='->', color='orange', lw=1, ls='--'))
    ax.text(0.95 * size, 0.04 * size, 'beam', color='orange', fontsize=8, ha='right', va='bottom')
    ax.set_xlim(-size, size)
    ax.set_ylim(-size, size)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('horizontal')
    ax.set_ylabel('vertical')
    ax.set_title('laboratory side view (Ys toward you)', fontsize=8)
    for ax, (i, j) in zip(axes[1:], ((0, 1), (1, 2))):
        ax.clear()
        for edge in edges:
            ax.plot(edge[:, i], edge[:, j], color=color, lw=1.2)
        for k, col in enumerate(('tab:red', 'tab:green', 'tab:blue')):
            ax.annotate('', xy=(basis[i, k], basis[j, k]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=col, lw=1.5))
        if abs(beam[i]) + abs(beam[j]) > 1e-6:
            ax.annotate('beam', xy=(0.9 * size * beam[i], 0.9 * size * beam[j]), xytext=(0, 0), color='orange',
                        arrowprops=dict(arrowstyle='->', color='orange', lw=1, ls='--'), fontsize=8)
        ax.set_xlim(-size, size)
        ax.set_ylim(-size, size)
        ax.set_aspect('equal', adjustable='box')
        names = ('Xs', 'Ys', 'Zs')
        ax.set_xlabel(names[i])
        ax.set_ylabel(names[j])
        ax.axhline(0, color='0.85', lw=0.6)
        ax.axvline(0, color='0.85', lw=0.6)
    axes[1].set_title('seen from above the sample (along Zs)', fontsize=8)
    axes[2].set_title('side view along Xs', fontsize=8)
    return basis


def describe_orientation(UB, latticeparams: Sequence[float], sample_tilt: float = 40.) -> str:
    """crystal directions along the sample normal Zs, along Xs and along the incoming beam"""
    basis = cell_basis_sample(UB, latticeparams, sample_tilt)
    beam = GT.matRot([0, 1, 0], -sample_tilt).T @ np.array([1., 0., 0.])
    parts = []
    for name, vec in (('Zs (normal)', [0, 0, 1]), ('Xs', [1, 0, 0]), ('beam', beam)):
        uvw = direction_indices(basis, vec)
        parts.append(f"{name} // [{uvw[0]:.2f} {uvw[1]:.2f} {uvw[2]:.2f}]")
    if is_hexagonal(latticeparams):
        cvec = basis[:, 2] / np.linalg.norm(basis[:, 2])
        tilt = np.degrees(np.arccos(abs(cvec[2])))
        parts.append(f'c axis {tilt:.1f} deg from the normal')
    return ';  '.join(parts)


def plot_cell_views(UB, latticeparams: Sequence[float], sample_tilt: float = 40., title: Optional[str] = None,
                    fig=None, color='tab:blue', view: Optional[Sequence[float]] = None):
    """figure: 3D view of the cell (BM32 laboratory view, see plot_cell(); view: (elevation, azimuth), default: the
    view of the 3D axes already in fig, e.g. rotated with the mouse, else DEFAULT_VIEW) and 3 projections"""
    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.figure(figsize=(15, 4.3))
    if view is None:   # keep the point of view of the previous drawing
        previous = [ax for ax in fig.axes if getattr(ax, 'name', '') == '3d']
        view = (previous[0].elev, previous[0].azim) if previous else None
    fig.clear()
    grid = fig.add_gridspec(1, 4, width_ratios=[1.35, 1, 1, 1], wspace=0.35)
    ax3d = fig.add_subplot(grid[0], projection='3d')
    plot_cell(ax3d, UB, latticeparams, sample_tilt, color=color, view=view)
    axes = [fig.add_subplot(grid[k]) for k in (1, 2, 3)]
    plot_cell_projections(axes, UB, latticeparams, sample_tilt, color=color)
    text = describe_orientation(UB, latticeparams, sample_tilt)
    fig.suptitle((title + '\n' if title else '') + text, fontsize=9)
    return fig


class CellBrowser:
    """unit cell of grains in the sample frame, one grain at a time (slider): 3D view and 3 projections

    :param grains: list of dict with keys 'UB' (3x3), 'material' (key of the LaueTools dict of materials) or
        'latticeparams', and optional 'label', 'color'
    """

    def __init__(self, grains: Sequence[Dict[str, Any]], sample_tilt: float = 40., dictmaterials=None):
        import ipywidgets as widgets
        import matplotlib.pyplot as plt
        from IPython.display import display
        self.grains = list(grains)
        self.sample_tilt = sample_tilt
        self.dictmaterials = dictmaterials
        materials = sorted({g.get('material', '') for g in self.grains})
        self.colors = {m: f'C{k}' for k, m in enumerate(materials)}
        self.slider = widgets.IntSlider(min=0, max=max(0, len(self.grains) - 1), value=0, description='grain',
                                        continuous_update=False, layout=widgets.Layout(width='60%'))
        self.label = widgets.HTML()
        self.widget_canvas = 'ipympl' in plt.get_backend() or 'widget' in plt.get_backend()
        if self.widget_canvas:
            with plt.ioff():
                self.fig = plt.figure(figsize=(15, 4.3))
            self.out = self.fig.canvas
        else:
            self.out = widgets.Output()
        self.slider.observe(lambda change: self.update(), names='value')
        display(widgets.VBox([self.slider, self.label, self.out]))
        self.update()

    def update(self):
        import matplotlib.pyplot as plt
        if not self.grains:
            self.label.value = 'no grain'
            return
        g = self.grains[self.slider.value]
        params = g.get('latticeparams') or lattice_parameters(g['material'], self.dictmaterials)
        title = g.get('label', f'grain {self.slider.value}')
        self.label.value = f'<b>{title}</b>'
        color = g.get('color', self.colors.get(g.get('material', ''), 'tab:blue'))
        if self.widget_canvas:   # the point of view (rotated with the mouse) is kept from grain to grain
            plot_cell_views(g['UB'], params, self.sample_tilt, title, fig=self.fig, color=color)
            self.fig.canvas.draw_idle()
        else:
            self.out.clear_output(wait=True)
            with self.out:
                plot_cell_views(g['UB'], params, self.sample_tilt, title, color=color, view=DEFAULT_VIEW)
                plt.show()


def grains_from_fitfiles(fitfiles: Sequence[str]) -> List[Dict[str, Any]]:
    """grains (dict UB, material, label) of .fit files, for CellBrowser"""
    try:
        from . import IOLaueTools as IOLT
    except ImportError:
        import IOLaueTools as IOLT
    import os
    grains = []
    for path in fitfiles:
        data = IOLT.readfitfile_multigrains(str(path), readmore=True)
        lines = open(path, errors='replace').read().splitlines()
        material = next((lines[k + 1].strip('# ').strip() for k, line in enumerate(lines[:-1])
                         if line.strip() == '#Element'), None)
        for k, ub in enumerate(np.reshape(np.ravel(data[3]), (-1, 9))):
            grains.append({'UB': ub.reshape(3, 3), 'material': material,
                           'label': f'{os.path.basename(str(path))} grain {k} ({material})'})
    return grains
