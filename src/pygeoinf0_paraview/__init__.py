"""pygeoinf0-paraview: functions on pygeoinf0's domains, split domains and direct sums as VTK files for ParaView (a plugin;
plan 056). Writing needs numpy only; ParaView (installed by the user) opens the files.

    report = export(f, "model.vtm", spacing=None, nodes=None, units=None)
        f: an element, a family or a list of elements of any space pg.port.fields takes (a function on a domain, pg.PiecewiseSpace,
        pg.PiecewiseSpace, a pg.DirectSum of function spaces). Writes model.vtm and, in the folder model/, one structured block per
        piece with its own nodes (a jump between pieces stays a jump: nothing is interpolated across a seam), one array per
        (field, member) named "field" or "field [member]", and the pieces' faces under "boundaries" (a seam written once).
        spacing: the largest Cartesian step along each internal axis (default: the largest piece's diameter / 64); nodes:
        nodes per axis instead (an int, or one per axis). units: {field: text}, stored with the arrays. surfaces: {summand:
        radius} draws summands on a sphere outside the partition (boundary heights) on the sphere of that radius.

    scene(report, array=None, plane=(polar, azimuth, offset), radius=None, clim="robust", screenshot=None)
        a ready ParaView script beside the export: plane and sphere slices with their widgets, boundary rings on the plane, a
        symmetric colour range; the colour menu is the member selector (scenes.py). open_in_paraview(report, ...) also starts
        ParaView, or says how to open the scene when ParaView is not found.

    html(f, "model.html", spacing=None, ...)   one self-contained page that cuts planes and spheres on the spot from each
        piece's own nodes, with no ParaView (html.py)

The report (also stored in each block's field data and in model.txt) states, per piece: the nodes and steps; the accuracy of
the node values (exact for formulas; a realisation's estimated error in its space's norm, which bounds no single value); the
error of ParaView's interpolation, measured at every cell centre against the exact part (largest and RMS; a measurement, not
a bound between centres); and how far the faceted cells are from the curved piece. A feature narrower than about two steps
can be missed between nodes: the report names the steps.
"""
import math
import os

import numpy as np
import pygeoinf0 as pg
from pygeoinf0.core.claims import CHECKED, Claim

from . import grid as G
from .html import html
from .scenes import open_in_paraview, paraview, scene
from .vtk_xml import write_vtm, write_vts

__version__ = "0.1.0"


class Report:
    """What an export wrote and what it cannot show (module docstring)."""

    def __init__(self, path):
        self.path, self.pieces, self.boundaries, self.absent = path, [], [], []
        self.centre, self.extent, self.surfaces = None, None, []

    def scale(self, array, clim="robust"):
        """The symmetric colour range of an array over all pieces: "robust" the 99th percentile of its nonzero |values| at
        the nodes (one strong piece does not pale the rest; colours saturate above), "whole" the largest |value|."""
        rows = [a for p in self.pieces for a in p["arrays"] if a["array"] == array]
        if not rows:
            raise KeyError(f"{array!r} is not an array of this export ({', '.join(self.arrays)})")
        top = max(a["p99"] if clim == "robust" else a["scale"] for a in rows)
        return (-top, top) if top > 0 else (-1.0, 1.0)

    @property
    def volume_arrays(self):
        """Arrays with values in every piece (not only NaN somewhere): the ones a resampled volume keeps meaningfully."""
        solid = [p for p in self.pieces if not p["name"].endswith("(surface)")]
        names = [a["array"] for a in solid[0]["arrays"]] if solid else []
        return [n for n in names if all(any(a["array"] == n for a in p["arrays"]) for p in solid)]

    @property
    def arrays(self):
        return list(dict.fromkeys(a["array"] for p in self.pieces for a in p["arrays"]))

    def lines(self):
        out = [f"{self.path}: {len(self.pieces)} piece(s), {len(self.boundaries)} boundary surface(s)"]
        for p in self.pieces:
            out.append(f"- {p['name']} on {p['domain']}: nodes {' x '.join(map(str, p['nodes']))}, largest step "
                       + ", ".join(f"{n} {s:.3g}" for n, s in zip(p["axes"], p["steps"]))
                       + f"; faceting up to {p['facet']:.2g}")
            for a in p["arrays"]:
                if a["scale"] == 0 and a["max"] == 0:
                    out.append(f"  - {a['array']}: zero at every node and cell centre")
                    continue
                rel = a["max"] / a["scale"] if a["scale"] > 0 else math.nan
                out.append(f"  - {a['array']}: values {a['accuracy']}; ParaView's interpolation at {a['centres']} cell centres: "
                           f"largest error {a['max']:.2g} ({rel:.2g} of max |value| {a['scale']:.3g}), RMS {a['rms']:.2g}")
            if p.get("undefined"):
                out.append(f"  - {', '.join(p['undefined'])}: not fields of this piece, written as NaN (transparent)")
        for n, why in self.absent:
            out.append(f"- {n}: not written ({why})")
        out.append("A feature narrower than about two steps can lie between nodes and not be shown; errors at cell centres are "
                   "measured, not bounded between them.")
        return out

    def __str__(self):
        return "\n".join(self.lines())

    __repr__ = __str__


def _array_name(field, member, many_members):
    return f"{field} [{member}]" if many_members else field


def _block(F, p, members, h, nodes, units, field_names=None, stretch=1.0, undefined=()):
    """Grid one piece p of F, its values per (field, member) and its error account: (shape, points, data, field data,
    report row, axes). field_names renames F's fields in the arrays; stretch scales the points (a surface given on the unit
    sphere drawn at its radius; the step h is then a step on the drawn surface)."""
    c = p.chart
    axes = G.axis_nodes(c, h / stretch, nodes)
    shape = tuple(len(a) for a in axes)
    X = c.to_cartesian(*np.meshgrid(*axes, indexing="ij")).reshape(-1, c.cartesian_dim)
    shown_at, true_at = G.cell_centres(c, axes)
    facet = stretch * float(np.max(np.linalg.norm(shown_at - true_at, axis=1))) if len(shown_at) else 0.0
    _, pts = G.grid(c, axes)
    data, fields_txt, arrays = {}, {}, []
    for fl in p.fields:
        label = (field_names or {}).get(fl, fl)
        for i, m in enumerate(members):
            name = _array_name(label, m, len(members) > 1)
            v = F.values(p, fl, i, X)
            # a volume cell is shown at the mean of its corners, which lies in the piece; a surface cell's mean lies off the
            # surface, so there the value is compared at the chart's own centre point (the offset is the faceting, stated)
            at = shown_at if c.dim == c.cartesian_dim else true_at
            mx, rms, n = G.interpolation_error(v, axes, F.values(p, fl, i, at))
            acc = F.accuracy(p, fl, i)
            fin = np.abs(v[np.isfinite(v)])
            scale = float(fin.max()) if fin.size else 0.0
            p99 = float(np.percentile(fin[fin > 0], 99)) if np.any(fin > 0) else 0.0
            arrays.append(dict(array=name, accuracy=acc["text"], kind=acc["kind"], max=mx, rms=rms, centres=n, scale=scale,
                               p99=p99, field=label))
            data[name] = G.to_vtk_order(v, shape)
            fields_txt[f"{name}: accuracy"] = acc["text"]
            fields_txt[f"{name}: interpolation"] = (f"largest error {mx:.3g}, RMS {rms:.3g} at {n} cell centres "
                                                    "(measured, not bounded between them)")
            if label in units:
                fields_txt[f"{name}: units"] = str(units[label])
    steps = [stretch * float(np.max(np.linalg.norm(np.diff(np.moveaxis(X.reshape(shape + (c.cartesian_dim,)), k, 0), axis=0),
                                                   axis=-1))) if shape[k] > 1 else 0.0 for k in range(c.dim)]
    fields_txt["sampling"] = (f"nodes {' x '.join(map(str, shape))} on the chart {c!r}"
                              + (f", drawn at {stretch:g} times its size" if stretch != 1.0 else "") + "; largest steps "
                              + ", ".join(f"{n} {s:.3g}" for n, s in zip(c.names, steps))
                              + f"; faceting of curved cells up to {facet:.3g}; features narrower than about two steps "
                                "can be missed")
    for fl in undefined:                                       # a field of the export that this piece does not carry
        for m in members:
            name = _array_name(fl, m, len(members) > 1)
            data[name] = np.full(int(np.prod(shape)), np.nan)
            fields_txt[f"{name}: not defined"] = (f"{fl} is not a field of the piece {p.name}: NaN here (transparent), so the "
                                                  "array exists in every block")
    row = dict(name=p.name, domain=repr(p.domain), nodes=shape, axes=c.names, steps=steps, facet=facet, arrays=arrays,
               undefined=list(undefined))
    return shape, stretch * pts, data, fields_txt, row, axes


def export(f, path, spacing=None, nodes=None, units=None, surfaces=None):
    """Write f for ParaView (module docstring); returns the Report. surfaces: {summand: radius} draws summands that live on a
    sphere outside the partition (a boundary's heights on the unit sphere) on the sphere of that radius, under "surfaces"."""
    F = pg.port.fields(f)
    units = units or {}
    path = os.path.abspath(os.path.expanduser(path if path.endswith(".vtm") else path + ".vtm"))
    stem = os.path.splitext(os.path.basename(path))[0]
    folder = os.path.join(os.path.dirname(path), stem)
    os.makedirs(folder, exist_ok=True)
    surfaces = dict(surfaces or {})
    solid = [p for p in F.pieces if p.name not in surfaces]          # a summand drawn as a surface is not also a piece
    if not solid:
        raise ValueError("nothing left to grid: every piece is drawn as a surface")
    diam = max(_diameter(p.chart) for p in solid)
    h = float(spacing) if spacing is not None else diam / 64.0
    report = Report(path)
    blocks, bounds, seen, surf = [], [], [], []
    for p in solid:
        shape, pts, data, fields_txt, row, axes = _block(F, p, F.members, h, nodes, units,
                                                         undefined=[fl for fl in F.fields if fl not in p.fields])
        fields_txt["piece"] = f"{p.name} on {p.domain!r}"
        fname = f"{_safe(p.name)}.vts"
        write_vts(os.path.join(folder, fname), shape, pts, data, fields_txt)
        blocks.append((p.name, f"{stem}/{fname}"))
        report.pieces.append(row)
        for label, fshape, fpts, key in G.faces(p.chart, axes, diam):
            dup = next((s for s in seen if s[3][0] == key[0] and np.allclose(s[3][1], key[1], atol=1e-12 * diam)), None)
            if dup is None:
                seen.append([[f"{p.name}: {label}"], fshape, fpts, key])
                continue
            dup[0].append(f"{p.name}: {label}")
            if np.prod(fshape) > np.prod(dup[1]):              # keep the finer of the two grids of a seam
                dup[1], dup[2] = fshape, fpts
    drawn = set()
    for key, radius in surfaces.items():
        S = pg.port.fields([u[key] for u in F.elements])
        if len(S.pieces) != 1 or type(S.pieces[0].domain).__name__ != "Sphere":
            raise ValueError(f"surfaces: {key!r} is not a function on a sphere ({S.pieces[0].domain!r})")
        q = S.pieces[0]
        stretch = float(radius) / float(q.domain.radius)
        shape, pts, data, fields_txt, row, _ = _block(S, q, F.members, h, nodes, units, {q.fields[0]: key}, stretch)
        fields_txt["surface"] = f"{key} on {q.domain!r}, drawn on the sphere of radius {radius:g}"
        fname = f"surface_{_safe(key)}.vts"
        write_vts(os.path.join(folder, fname), shape, pts, data, fields_txt)
        surf.append((key, f"{stem}/{fname}"))
        row["name"], row["domain"] = f"{key} (surface)", f"{q.domain!r} drawn at radius {radius:g}"
        report.pieces.append(row)
        drawn.add(key)
    report.absent = [(n, why) for n, why in F.absent if n not in drawn]
    for i, (labels, fshape, fpts, _) in enumerate(seen):
        fname = f"boundary_{i}.vts"
        write_vts(os.path.join(folder, fname), fshape, fpts, {}, {"boundary": " | ".join(labels)})
        bounds.append((" | ".join(labels), f"{stem}/{fname}"))
        report.boundaries.append(" | ".join(labels))
    allpts = np.vstack([_corner_points(p.chart) for p in solid])
    lo, hi = allpts.min(axis=0), allpts.max(axis=0)
    report.centre = [float(c) for c in 0.5 * (lo + hi)] + [0.0] * (3 - len(lo))
    report.extent = float(0.5 * np.linalg.norm(hi - lo))
    report.surfaces = [k for k, _ in surf]
    write_vtm(path, [("pieces", blocks)] + ([("boundaries", bounds)] if bounds else []) + ([("surfaces", surf)] if surf else []))
    with open(os.path.join(os.path.dirname(path), stem + ".txt"), "w") as fh:
        fh.write(str(report) + "\n")
    return report


def _diameter(chart):
    t = [np.linspace(a, b, 9) for a, b in chart.reference]
    X = chart.to_cartesian(*np.meshgrid(*t, indexing="ij")).reshape(-1, chart.cartesian_dim)
    return float(np.linalg.norm(X.max(axis=0) - X.min(axis=0)))


def _corner_points(chart):
    t = [np.linspace(a, b, 9) for a, b in chart.reference]
    return chart.to_cartesian(*np.meshgrid(*t, indexing="ij")).reshape(-1, chart.cartesian_dim)


def _safe(name):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name)) or "piece"


PLUGIN = pg.port.plugins.Plugin(
    name="paraview", version=__version__, port=">=1,<2",
    objects={"export": export, "scene": scene, "open_in_paraview": open_in_paraview, "html": html},
    claims=(Claim("each piece is written on its own nodes, so values at a seam are that piece's own limit", CHECKED,
                  "tests/test_export.py: read back and compared with the exact parts on both sides of a seam", key="pieces"),),
    licence="BSD-3-Clause", source="github.com/Adrian-Mag/pygeoinf0-paraview-public")
pg.port.plugins.register(PLUGIN)

__all__ = ["PLUGIN", "Report", "export", "html", "open_in_paraview", "paraview", "scene"]
