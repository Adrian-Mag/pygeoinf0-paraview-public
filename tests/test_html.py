"""pygeoinf0-paraview's portable page (plan 056 P4). The page's own JavaScript is run in Node.js (skipped without it) and its
values compared with the exact parts on both sides of a seam; the report's interpolation error bounds what it shows at cell
centres."""
import json
import os
import shutil
import subprocess
import textwrap

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0_paraview import html

R1 = 0.55
IN = lambda x, y, z: 1 + 0.2 * z + 0.3 * np.sin(3 * x)       # noqa: E731
OUT = lambda x, y, z: -1 + 0.2 * z + 0.3 * np.cos(2 * y)     # noqa: E731
NODE = shutil.which("node")


def _layered():
    V = pg.PiecewiseSpace.lebesgue(pg.LayeredBall([R1, 1.0], names=["core", "mantle"]))
    return V({"core": IN, "mantle": OUT}, name="m")


def _page_values(path, name, points):
    """Values the page's value() gives at points, run in Node with a stub DOM."""
    driver = textwrap.dedent("""
        const fs = require("fs"), vm = require("vm");
        const [file, name, pts] = process.argv.slice(-3);
        const page = fs.readFileSync(file, "utf8");
        const js = page.slice(page.indexOf("<script>") + 8, page.lastIndexOf("</script>"));
        const el = () => ({ value: "0", textContent: "", hidden: false, width: 0, height: 0, appendChild() {}, addEventListener() {},
            getBoundingClientRect: () => ({ left: 0, top: 0, width: 1, height: 1 }),
            getContext: () => ({ createImageData: (w, h) => ({ data: new Uint8ClampedArray(4 * w * h) }), putImageData() {},
                                 beginPath() {}, arc() {}, stroke() {} }) });
        const els = {}; const document = { getElementById: id => (els[id] = els[id] || el()), createElement: el };
        const ctx = { document, atob: s => Buffer.from(s, "base64").toString("binary"), Math, Float32Array, Float64Array,
                      Uint8Array, Uint8ClampedArray, JSON, isNaN, console };
        vm.createContext(ctx);
        els.kind = Object.assign(el(), { value: "plane" }); els.pol = Object.assign(el(), { value: "90" });
        els.array = Object.assign(el(), { value: name });
        vm.runInContext(js, ctx);
        const P = JSON.parse(pts);
        console.log(JSON.stringify(P.map(p => { const v = ctx.value(p[0], p[1], p[2], name)[0]; return isNaN(v) ? null : v; })));
        """)
    out = subprocess.run([NODE, "-e", driver, "--", str(path), name, json.dumps(np.asarray(points).tolist())],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    return np.array([np.nan if v is None else v for v in json.loads(out.stdout.strip().splitlines()[-1])])


def test_the_page_states_its_storage_and_errors(tmp_path):
    rep = html(_layered(), str(tmp_path / "m.html"), spacing=0.05)
    text = open(rep["path"]).read()
    assert "float32" in rep["report"] and "measured, not bounded" in rep["report"]
    assert "this page's interpolation" in text and len(rep["pieces"]) == 2


def test_other_domains_are_refused_by_name(tmp_path):
    with pytest.raises(pg.errors.Unsupported, match="balls, shells and 3-D boxes"):
        html(pg.L2(pg.Box((0, 1), (0, 1))).element(lambda x, y: x), str(tmp_path / "b.html"))


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_the_page_keeps_the_jump_and_its_values_within_the_reported_error(tmp_path):
    rep = html(_layered(), str(tmp_path / "m.html"), spacing=0.05)
    u = np.array([1.0, 0.3, 0.2]) / np.linalg.norm([1.0, 0.3, 0.2])
    seam = [0.549 * u, 0.551 * u]
    got = _page_values(rep["path"], "value", seam)
    np.testing.assert_allclose(got, [IN(*seam[0]), OUT(*seam[1])], atol=2e-3)
    assert got[0] - got[1] > 1.9
    rng = np.random.default_rng(1)
    d = rng.normal(size=(300, 3))
    P = d / np.linalg.norm(d, axis=1, keepdims=True) * rng.uniform(0.05, 0.99, (300, 1)) ** (1 / 3)
    r = np.linalg.norm(P, axis=1)
    exact = np.where(r < R1, IN(*P.T), OUT(*P.T))
    worst = max(a["max"] for p in rep["pieces"] for a in p["arrays"])
    err = np.abs(_page_values(rep["path"], "value", P) - exact)
    assert np.all(np.isfinite(err)) and err.max() <= 2 * worst, (err.max(), worst)
    assert np.isnan(_page_values(rep["path"], "value", [[1.2, 0, 0]])[0])            # outside every piece


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_the_page_draws_without_errors(tmp_path):
    rep = html([_layered(), 2 * _layered()], str(tmp_path / "m.html"), spacing=0.1, pixels=64)
    _page_values(rep["path"], "value [m]", [[0.1, 0.0, 0.0]])            # runs draw() on load as well
