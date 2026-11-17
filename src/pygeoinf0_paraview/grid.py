"""Grids on a piece's chart and what ParaView's interpolation of them misses (plan 056 P2).

Nodes form a tensor grid in the chart's reference box, with every break and both ends of every axis as nodes, mapped by
to_cartesian: the grid conforms to the piece (a shell's nodes lie on both of its spheres). ParaView interpolates a structured
cell multilinearly in its parametric coordinates, and places the cell by the same multilinear map of its corners, so at a
cell's parametric centre it shows the mean of the corner values at the mean of the corner points. measure() compares that
with the exact value of the piece's part at that point: a measured error at every cell centre, not a bound elsewhere. The
faceting of curved pieces is measured too: the distance between the chart's image of a cell's reference centre and the mean
of its corners.
"""
import math

import numpy as np

PROBE = 5          # points per other axis when the length of an axis is measured
LINE = 65          # points along a probe line


def _lengths(chart):
    """Largest Cartesian length of each internal axis: polylines of LINE points along it, at PROBE points of the others."""
    out = []
    for k, (lo, hi) in enumerate(chart.reference):
        axes = [np.linspace(a, b, LINE if j == k else PROBE) for j, (a, b) in enumerate(chart.reference)]
        X = chart.to_cartesian(*np.meshgrid(*axes, indexing="ij"))
        X = np.moveaxis(X, k, 0)
        seg = np.linalg.norm(np.diff(X, axis=0), axis=-1).sum(axis=0)
        out.append(float(seg.max()))
    return out


def axis_nodes(chart, spacing=None, nodes=None):
    """Nodes per axis: every break and both ends; between them uniform in the reference coordinate, with at most `spacing`
    of Cartesian length per step along the axis's longest line (or `nodes` per axis, as given: an int or one per axis)."""
    L = _lengths(chart)
    out = []
    for k, (lo, hi) in enumerate(chart.reference):
        knots = [lo, *chart.breaks[k], hi]
        if nodes is not None:
            total = int(nodes[k] if np.ndim(nodes) else nodes)
        else:
            total = 1 + max(1, math.ceil(L[k] / spacing)) if L[k] > 0 else 2
        steps = max(total - 1, len(knots) - 1)
        span = hi - lo
        t = [lo]
        for a, b in zip(knots[:-1], knots[1:]):
            m = max(1, round(steps * (b - a) / span))
            t.extend(np.linspace(a, b, m + 1)[1:])
        out.append(np.asarray(t, float))
    return out


def grid(chart, axes):
    """(shape, points (N, 3) with x fastest as VTK wants: axis 0 fastest)."""
    T = np.meshgrid(*axes, indexing="ij")
    X = chart.to_cartesian(*T)                                  # (n0, n1, ..., cdim)
    pts = np.zeros(X.shape[:-1] + (3,))
    pts[..., :chart.cartesian_dim] = X
    # VTK order: index 0 fastest; numpy C order makes the last axis fastest, so transpose the grid axes
    pts = np.transpose(pts, tuple(range(chart.dim))[::-1] + (chart.dim,))
    return tuple(len(a) for a in axes), pts.reshape(-1, 3)


def to_vtk_order(values, shape):
    """Node values in meshgrid ("ij") C order -> VTK order (axis 0 fastest)."""
    v = np.asarray(values).reshape(shape)
    return np.transpose(v, tuple(range(len(shape)))[::-1]).ravel()


def _corners(A):
    """Mean over the 2^d corners of every cell of an array on the node grid ("ij" layout, one axis per grid axis)."""
    out = A
    for k in range(A.ndim):
        sl0 = [slice(None)] * out.ndim
        sl1 = [slice(None)] * out.ndim
        sl0[k], sl1[k] = slice(None, -1), slice(1, None)
        out = 0.5 * (out[tuple(sl0)] + out[tuple(sl1)])
    return out


def cell_centres(chart, axes):
    """(points shown by ParaView at the cells' parametric centres, the chart's own centre points), both (cells, cdim)."""
    T = np.meshgrid(*axes, indexing="ij")
    X = chart.to_cartesian(*T)
    shown = np.stack([_corners(X[..., c]) for c in range(chart.cartesian_dim)], axis=-1)
    mids = [0.5 * (a[:-1] + a[1:]) for a in axes]
    true = chart.to_cartesian(*np.meshgrid(*mids, indexing="ij"))
    return shown.reshape(-1, chart.cartesian_dim), true.reshape(-1, chart.cartesian_dim)


def interpolation_error(node_values, axes, exact_at_centres):
    """(largest |shown - exact| at cell centres, its RMS, the number of centres compared) for node values in "ij" layout."""
    shown = _corners(np.asarray(node_values, float).reshape(tuple(len(a) for a in axes))).ravel()
    e = np.abs(shown - np.asarray(exact_at_centres, float).ravel())
    ok = np.isfinite(e)
    if not ok.any():
        return math.nan, math.nan, 0
    return float(e[ok].max()), float(np.sqrt(np.mean(e[ok] ** 2))), int(ok.sum())


def faces(chart, axes, scale):
    """The faces of the piece at both ends of every non-periodic axis that have area (a ball's centre and a pole do not):
    [(label, shape, points, key)] as 2-D structured grids (for a 2-D piece: its edge curves; none for 1-D). key: the face's
    reference ranges and its image at 3 points per axis of them, which equal for two pieces' faces when they parametrise the
    same surface in the same way (a seam)."""
    if chart.dim < 2:
        return []
    out = []
    for k in range(chart.dim):
        if chart.periodic[k]:
            continue
        for end, value in (("lo", chart.reference[k][0]), ("hi", chart.reference[k][1])):
            sub = [np.array([value]) if j == k else a for j, a in enumerate(axes)]
            T = np.meshgrid(*sub, indexing="ij")
            X = np.squeeze(chart.to_cartesian(*T), axis=k)            # (.., cdim) over the other axes
            spans = []
            for j in range(X.ndim - 1):
                Y = np.moveaxis(X, j, 0)
                spans.append(float(np.linalg.norm(np.diff(Y, axis=0), axis=-1).sum(axis=0).max()) if Y.shape[0] > 1 else 0.0)
            if min(spans) <= 1e-12 * scale:
                continue
            others = [a for j, a in enumerate(axes) if j != k]
            pts = np.zeros(X.shape[:-1] + (3,))
            pts[..., :chart.cartesian_dim] = X
            pts = np.transpose(pts, tuple(range(X.ndim - 1))[::-1] + (X.ndim - 1,)).reshape(-1, 3)
            rng = [r for j, r in enumerate(chart.reference) if j != k]
            probe = [np.linspace(a, b, 3) for a, b in rng]
            P = np.meshgrid(*[np.array([value]) if j == k else probe[j - (j > k)] for j in range(chart.dim)], indexing="ij")
            key = (tuple(rng), np.asarray(chart.to_cartesian(*P), float).reshape(-1, chart.cartesian_dim))
            out.append((f"{chart.names[k]} = {value:.6g}", tuple(len(a) for a in others), pts, key))
    return out
