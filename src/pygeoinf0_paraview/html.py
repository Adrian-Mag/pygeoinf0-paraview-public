"""A portable view: one self-contained HTML file that cuts planes and spheres on the spot from each piece's own nodes, so a jump
at a seam stays a jump, with no ParaView and no Python to reopen it (plan 056 P4).

    report = html(f, "model.html", spacing=None, nodes=None, units=None, title=None, pixels=360)

f as for export (pieces of pg.PiecewiseSpace, a pg.DirectSum, or one domain): pieces on balls and shells (a spherical
chart about their centre) or 3-D boxes; others are refused by name, since the page must invert their charts. The page stores
each piece's reference axes and its node values in float32 (rounding about 6e-8 of a value, stated) and, for each pixel, finds
the piece containing the point and interpolates trilinearly in that piece's reference coordinates (r, polar, azimuth or x, y,
z). The report (shown on the page and returned) measures that interpolation at every cell's reference centre against the
exact part (largest and RMS: measured, not bounded between centres), with the nodes and steps per piece. Controls: plane
(polar, azimuth, offset) or sphere (radius) as a latitude-longitude map, field and member, a symmetric colour range ("robust",
"whole" or given), hover for the value and the piece.
"""
import base64
import json
import os

import numpy as np
import pygeoinf0 as pg

from . import grid as G


def _kind(domain):
    k = type(domain).__name__
    if k in ("Ball", "Shell"):
        return "spherical", [float(c) for c in domain.centre]
    if k == "Box" and domain.dim == 3:
        return "box", [0.0, 0.0, 0.0]
    raise pg.errors.Unsupported(f"the HTML view cuts balls, shells and 3-D boxes, not {domain!r}: use export() for ParaView")


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a, "<f4").tobytes()).decode("ascii")


def html(f, path, spacing=None, nodes=None, units=None, title=None, pixels=360, clim="robust"):
    """Write the portable view (module docstring); returns its report as a dict."""
    from . import _diameter, _array_name
    F = pg.port.fields(f)
    units = units or {}
    path = os.path.abspath(os.path.expanduser(path if path.endswith(".html") else path + ".html"))
    diam = max(_diameter(p.chart) for p in F.pieces)
    h = float(spacing) if spacing is not None else diam / 48.0
    pieces, rows, lim = [], [], {}
    for p in F.pieces:
        kind, centre = _kind(p.domain)
        c = p.chart
        axes = G.axis_nodes(c, h, nodes)
        shape = tuple(len(a) for a in axes)
        X = c.to_cartesian(*np.meshgrid(*axes, indexing="ij")).reshape(-1, 3)
        mids = [0.5 * (a[:-1] + a[1:]) for a in axes]
        Xm = c.to_cartesian(*np.meshgrid(*mids, indexing="ij")).reshape(-1, 3)
        arrays, row = {}, dict(name=p.name, domain=repr(p.domain), nodes=shape, arrays=[])
        for fl in p.fields:
            for i, m in enumerate(F.members):
                name = _array_name(fl, m, len(F.members) > 1)
                v = F.values(p, fl, i, X)
                if np.all(v == 0):                                  # stated in the report, not stored
                    row["arrays"].append(dict(array=name, zero=True))
                    continue
                v32 = v.astype(np.float32)
                mx, rms, n = G.interpolation_error(v32.astype(float), axes, F.values(p, fl, i, Xm))
                acc = F.accuracy(p, fl, i)
                fin = np.abs(v[np.isfinite(v)])
                top = float(np.percentile(fin[fin > 0], 99) if clim == "robust" else fin.max()) if np.any(fin > 0) else 0.0
                lim[name] = max(lim.get(name, 0.0), top)
                arrays[name] = _b64(v32)
                row["arrays"].append(dict(array=name, accuracy=acc["text"], max=mx, rms=rms, centres=n,
                                          scale=float(fin.max()) if fin.size else 0.0, units=units.get(fl, "")))
        steps = [float(np.max(np.linalg.norm(np.diff(np.moveaxis(X.reshape(shape + (3,)), k, 0), axis=0), axis=-1)))
                 for k in range(3)]
        row["steps"] = dict(zip(c.names, steps))
        rows.append(row)
        pieces.append(dict(name=p.name, kind=kind, centre=centre, axes=[a.tolist() for a in axes], shape=shape,
                           periodic=list(c.periodic), arrays=arrays))
    if not isinstance(clim, str):
        lim = {k: max(abs(float(clim[0])), abs(float(clim[1]))) for k in lim}
    spherical = [p for p in pieces if p["kind"] == "spherical"]
    if spherical:
        centre, extent = spherical[0]["centre"], max(p["axes"][0][-1] for p in spherical)
    else:
        lo = np.array([[p["axes"][k][0] for k in range(3)] for p in pieces]).min(axis=0)
        hi = np.array([[p["axes"][k][-1] for k in range(3)] for p in pieces]).max(axis=0)
        centre, extent = [float(v) for v in 0.5 * (lo + hi)], float(0.5 * np.linalg.norm(hi - lo))
    radii = sorted({round(r, 12) for p in spherical for r in (p["axes"][0][0], p["axes"][0][-1]) if r > 0})
    report_lines = [f"{os.path.basename(path)}: {len(pieces)} piece(s); values stored in float32 (rounding about 6e-8 of a value)"]
    for r in rows:
        report_lines.append(f"{r['name']} on {r['domain']}: nodes {' x '.join(map(str, r['nodes']))}, largest steps "
                            + ", ".join(f"{k} {v:.3g}" for k, v in r["steps"].items()))
        for a in r["arrays"]:
            if a.get("zero"):
                report_lines.append(f"  {a['array']}: zero at every node (not stored; the page shows it as missing)")
                continue
            report_lines.append(f"  {a['array']}: {a['accuracy']}; this page's interpolation at {a['centres']} cell centres: "
                                f"largest error {a['max']:.2g} ({a['max'] / a['scale']:.2g} of max |value| {a['scale']:.3g}), "
                                f"RMS {a['rms']:.2g}")
    for n, why in F.absent:
        report_lines.append(f"{n}: not shown ({why})")
    report_lines.append("A feature narrower than about two steps can lie between nodes and not be shown; errors at cell centres "
                        "are measured, not bounded between them.")
    data = dict(title=title or os.path.splitext(os.path.basename(path))[0], pieces=pieces, centre=centre, extent=extent,
                radii=radii, lim=lim, arrays=list(lim), pixels=int(pixels), report=report_lines,
                units={a["array"]: a.get("units", "") for r in rows for a in r["arrays"]})
    with open(path, "w") as fh:
        fh.write(PAGE.replace("__DATA__", json.dumps(data)).replace("__TITLE__", data["title"]))
    return dict(path=path, pieces=rows, report="\n".join(report_lines), absent=F.absent)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --bg: #ffffff; --fg: #1d1d1f; --muted: #6e6e73; --line: #d2d2d7; --panel: #f5f5f7; }
@media (prefers-color-scheme: dark) { :root { --bg: #161618; --fg: #f2f2f2; --muted: #a1a1a6; --line: #3a3a3c; --panel: #222225; } }
body { margin: 0; padding: 16px; background: var(--bg); color: var(--fg); font: 14px/1.4 -apple-system, system-ui, sans-serif; }
h1 { font-size: 18px; margin: 0 0 12px; }
.wrap { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }
.controls { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; min-width: 260px; flex: 0 1 300px; }
.controls label { display: block; margin: 8px 0 2px; color: var(--muted); font-size: 12px; }
.controls input[type=range], .controls select { width: 100%; }
canvas { max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 8px; image-rendering: pixelated; background: var(--panel); }
#readout { font-variant-numeric: tabular-nums; min-height: 1.4em; color: var(--muted); margin-top: 6px; }
#bar { width: 100%; height: 12px; border-radius: 3px; margin-top: 4px; }
.ticks { display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; }
pre { white-space: pre-wrap; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; font-size: 12px; max-width: 100%; overflow-x: auto; }
</style></head><body>
<h1 id="title"></h1>
<div class="wrap">
 <div><canvas id="view"></canvas><div id="readout">hover for the value</div></div>
 <div class="controls">
  <label>view</label><select id="kind"><option value="plane">plane</option><option value="sphere">sphere (latitude-longitude map)</option></select>
  <label>array (field and member)</label><select id="array"></select>
  <div id="planeCtl"><label>polar <span id="polV"></span>°</label><input id="pol" type="range" min="0" max="180" step="1" value="90">
   <label>azimuth <span id="azV"></span>°</label><input id="az" type="range" min="0" max="360" step="1" value="0">
   <label>offset <span id="offV"></span></label><input id="off" type="range" min="-1" max="1" step="0.005" value="0"></div>
  <div id="sphereCtl" hidden><label>radius <span id="radV"></span></label><input id="rad" type="range" min="0" max="1" step="0.001" value="0.5"></div>
  <label>colour range ± <span id="limV"></span></label><input id="lim" type="range" min="-3" max="1" step="0.01" value="0">
  <canvas id="bar" width="256" height="12"></canvas><div class="ticks"><span id="lo"></span><span>0</span><span id="hi"></span></div>
 </div>
</div>
<pre id="report"></pre>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
$("title").textContent = D.title; $("report").textContent = D.report.join("\n");
function f32(b64) { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new Float32Array(u.buffer); }
for (const p of D.pieces) { p.vals = {}; for (const k in p.arrays) p.vals[k] = f32(p.arrays[k]); p.lo = p.axes.map(a => a[0]); p.hi = p.axes.map(a => a[a.length - 1]); }
for (const a of D.arrays) { const o = document.createElement("option"); o.value = o.textContent = a; $("array").appendChild(o); }
const N = D.pixels, cv = $("view"); cv.width = cv.height = N; const ctx = cv.getContext("2d"), img = ctx.createImageData(N, N);
function ref(p, x, y, z) {
  if (p.kind === "box") return [x, y, z];
  const dx = x - p.centre[0], dy = y - p.centre[1], dz = z - p.centre[2], r = Math.hypot(dx, dy, dz);
  let ph = Math.atan2(dy, dx); if (ph < 0) ph += 2 * Math.PI;
  return [r, r > 0 ? Math.acos(Math.max(-1, Math.min(1, dz / r))) : 0, ph];
}
function cell(a, t) { let lo = 0, hi = a.length - 1; if (t <= a[0]) return [0, 0]; if (t >= a[hi]) return [hi - 1, 1];
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (a[m] <= t) lo = m; else hi = m; } return [lo, (t - a[lo]) / (a[lo + 1] - a[lo])]; }
const TOL = 1e-12;
function value(x, y, z, name) {
  for (const p of D.pieces) {
    const v = p.vals[name]; if (!v) continue;
    const q = ref(p, x, y, z); let inside = true;
    for (let k = 0; k < 3; k++) { const s = TOL * Math.max(1, Math.abs(p.hi[k])); if (q[k] < p.lo[k] - s || q[k] > p.hi[k] + s) { inside = false; break; } }
    if (!inside) continue;
    const [i, a] = cell(p.axes[0], q[0]), [j, b] = cell(p.axes[1], q[1]), [k, c] = cell(p.axes[2], q[2]);
    const n1 = p.shape[1], n2 = p.shape[2], at = (I, J, K) => v[(I * n1 + J) * n2 + K];
    let s = 0;
    for (const [I, wi] of [[i, 1 - a], [i + 1, a]]) for (const [J, wj] of [[j, 1 - b], [j + 1, b]]) for (const [K, wk] of [[k, 1 - c], [k + 1, c]])
      { const w = wi * wj * wk; if (w) s += w * at(I, J, K); }
    return [s, p.name];
  }
  return [NaN, null];
}
function colour(t) {   // diverging: blue - white - red, t in [-1, 1]
  t = Math.max(-1, Math.min(1, t)); const u = Math.abs(t);
  return t < 0 ? [255 * (1 - 0.75 * u), 255 * (1 - 0.55 * u), 255 - 40 * u] : [255 - 30 * u, 255 * (1 - 0.7 * u), 255 * (1 - 0.75 * u)];
}
let points = null;
function geometry() {
  const P = new Float64Array(N * N * 3), c = D.centre, R = D.extent;
  if ($("kind").value === "plane") {
    const pol = $("pol").value * Math.PI / 180, az = $("az").value * Math.PI / 180, off = $("off").value * R;
    const n = [Math.sin(pol) * Math.cos(az), Math.sin(pol) * Math.sin(az), Math.cos(pol)];
    let e1 = Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    let d = e1[0] * n[0] + e1[1] * n[1] + e1[2] * n[2]; e1 = e1.map((v, k) => v - d * n[k]); d = Math.hypot(...e1); e1 = e1.map(v => v / d);
    const e2 = [n[1] * e1[2] - n[2] * e1[1], n[2] * e1[0] - n[0] * e1[2], n[0] * e1[1] - n[1] * e1[0]];
    for (let r = 0; r < N; r++) for (let q = 0; q < N; q++) { const u = (2 * q / (N - 1) - 1) * R, w = (1 - 2 * r / (N - 1)) * R, o = 3 * (r * N + q);
      for (let k = 0; k < 3; k++) P[o + k] = c[k] + off * n[k] + u * e1[k] + w * e2[k]; }
    return { P, n, off };
  }
  const rad = $("rad").value * R;
  for (let r = 0; r < N; r++) for (let q = 0; q < N; q++) { const lat = (90 - 180 * r / (N - 1)) * Math.PI / 180, lon = (360 * q / (N - 1) - 180) * Math.PI / 180, o = 3 * (r * N + q);
    P[o] = c[0] + rad * Math.cos(lat) * Math.cos(lon); P[o + 1] = c[1] + rad * Math.cos(lat) * Math.sin(lon); P[o + 2] = c[2] + rad * Math.sin(lat); }
  return { P, rad };
}
function draw() {
  const name = $("array").value, L = (D.lim[name] || 1) * Math.pow(10, +$("lim").value), g = geometry(); points = g;
  $("polV").textContent = $("pol").value; $("azV").textContent = $("az").value;
  $("offV").textContent = (+$("off").value * D.extent).toPrecision(3); $("radV").textContent = (+$("rad").value * D.extent).toPrecision(4);
  const u = D.units[name] ? " " + D.units[name] : ""; $("limV").textContent = L.toPrecision(3) + u; $("lo").textContent = (-L).toPrecision(3); $("hi").textContent = L.toPrecision(3);
  for (let i = 0; i < N * N; i++) { const [v] = value(g.P[3 * i], g.P[3 * i + 1], g.P[3 * i + 2], name), o = 4 * i;
    if (isNaN(v)) { img.data[o + 3] = 0; continue; } const [r, gg, b] = colour(v / L); img.data[o] = r; img.data[o + 1] = gg; img.data[o + 2] = b; img.data[o + 3] = 255; }
  ctx.putImageData(img, 0, 0);
  if ($("kind").value === "plane") {    // the seams of round pieces cut by the plane
    const d = g.off; ctx.strokeStyle = "rgba(0,0,0,0.8)"; ctx.lineWidth = 1;
    for (const r of D.radii) if (r > Math.abs(d)) { ctx.beginPath(); ctx.arc((N - 1) / 2, (N - 1) / 2, Math.sqrt(r * r - d * d) / D.extent * (N - 1) / 2, 0, 2 * Math.PI); ctx.stroke(); }
  }
}
function bar() { const b = $("bar").getContext("2d"), im = b.createImageData(256, 12);
  for (let q = 0; q < 256; q++) { const [r, g, bb] = colour(2 * q / 255 - 1); for (let y = 0; y < 12; y++) { const o = 4 * (y * 256 + q); im.data[o] = r; im.data[o + 1] = g; im.data[o + 2] = bb; im.data[o + 3] = 255; } }
  b.putImageData(im, 0, 0); }
cv.addEventListener("mousemove", e => { if (!points) return; const b = cv.getBoundingClientRect();
  const q = Math.floor((e.clientX - b.left) / b.width * N), r = Math.floor((e.clientY - b.top) / b.height * N); if (q < 0 || r < 0 || q >= N || r >= N) return;
  const i = 3 * (r * N + q), [v, piece] = value(points.P[i], points.P[i + 1], points.P[i + 2], $("array").value);
  $("readout").textContent = piece ? `${v.toPrecision(5)}${D.units[$("array").value] ? " " + D.units[$("array").value] : ""}  (piece ${piece}; point ${[0, 1, 2].map(k => points.P[i + k].toPrecision(4)).join(", ")})` : "outside the pieces"; });
$("kind").addEventListener("change", () => { $("planeCtl").hidden = $("kind").value !== "plane"; $("sphereCtl").hidden = $("kind").value === "plane"; draw(); });
for (const id of ["array", "pol", "az", "off", "rad", "lim"]) $(id).addEventListener("input", draw);
bar(); draw();
</script></body></html>
"""
