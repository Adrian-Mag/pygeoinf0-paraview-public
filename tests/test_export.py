"""pygeoinf0-paraview export (plan 056 P2). Exact answers: the formulas the elements were built from. No GUI: files are read
back with the plugin's own reader; one test runs ParaView's pvbatch when it is installed."""
import os
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0_paraview import export
from pygeoinf0_paraview.vtk_xml import read_vts

R1 = 0.55
IN = lambda x, y, z: 1 + 0.2 * z + 0.3 * np.sin(3 * x)       # noqa: E731
OUT = lambda x, y, z: -1 + 0.2 * z + 0.3 * np.cos(2 * y)     # noqa: E731
PVBATCH = shutil.which("pvbatch") or next((p for p in ("/Applications/ParaView-6.2.0.app/Contents/bin/pvbatch",)
                                           if os.path.exists(p)), None)


def _layered():
    P = pg.LayeredBall([R1, 1.0], names=["core", "mantle"])
    V = pg.PiecewiseSpace.lebesgue(P)
    return V({"core": IN, "mantle": OUT}, name="m")


def _read(folder, name):
    return read_vts(os.path.join(folder, name))


def test_each_piece_is_written_on_its_own_nodes_with_exact_values(tmp_path):
    rep = export(_layered(), str(tmp_path / "m.vtm"), spacing=0.08)
    for piece, f in (("core", IN), ("mantle", OUT)):
        shape, pts, data, fields = _read(tmp_path / "m", f"{piece}.vts")
        np.testing.assert_allclose(data["value"], f(*pts.T), atol=1e-12)
        r = np.linalg.norm(pts, axis=1)
        assert np.any(np.isclose(r, R1)), "the seam is a set of nodes of both pieces"
        assert "nodes" in fields["sampling"] and "accuracy" in "".join(fields)
    assert len(rep.boundaries) == 2 and "core: r = 0.55 | mantle: r = 0.55" in rep.boundaries
    assert "measured, not bounded" in str(rep)
    assert os.path.exists(tmp_path / "m.txt")


def test_the_report_finds_the_blur_of_a_grid_across_a_jump(tmp_path):
    """Planted control: the same field as one function on the whole ball, gridded across the jump, must report an error of
    the size of the jump; the pieces' own grids report about 1e-3."""
    good = export(_layered(), str(tmp_path / "good.vtm"), spacing=0.05)
    worst = max(a["max"] for p in good.pieces for a in p["arrays"])
    B = pg.L2(pg.Ball(1.0))
    glued = B.element(lambda x, y, z: np.where(x * x + y * y + z * z < R1 * R1, IN(x, y, z), OUT(x, y, z)), name="g",
                      breaks=None)
    bad = export(glued, str(tmp_path / "bad.vtm"), spacing=0.05)
    blur = bad.pieces[0]["arrays"][0]["max"]
    assert worst < 2e-3 and blur > 0.5, (worst, blur)


def test_the_interpolation_error_falls_as_the_square_of_the_step(tmp_path):
    e = [max(a["max"] for p in export(_layered(), str(tmp_path / f"h{i}.vtm"), spacing=h).pieces for a in p["arrays"])
         for i, h in enumerate((0.1, 0.05))]
    assert 2.5 < e[0] / e[1] < 6, e


def test_members_fields_units_and_absent_pieces(tmp_path):
    part = pg.Partition(pg.Ball(1.0), [pg.Ball(R1), pg.Shell(R1, 1.0)], names=["in", "out"])
    M = pg.PiecewiseSpace(part, {"vp": pg.L2, "vs": pg.L2}, where={"vs": ["out"]})
    m = M.split({"vp": lambda x, y, z: 1.0 + x, "vs": lambda x, y, z: 2.0 + y}, name="m")
    export([m, 2 * m], str(tmp_path / "op.vtm"), spacing=0.15, units={"vs": "km/s"})
    _, pts, data, fields = _read(tmp_path / "op", "out.vts")
    assert set(data) == {"vp [m]", "vp [2·m]", "vs [m]", "vs [2·m]"}
    np.testing.assert_allclose(data["vs [2·m]"], 2 * (2.0 + pts[:, 1]), atol=1e-12)
    assert fields["vs [m]: units"] == "km/s"
    _, _, data_in, fields_in = _read(tmp_path / "op", "in.vts")
    assert set(data_in) == set(data)                           # every array in every block: vs is NaN where it is no field
    assert np.all(np.isnan(data_in["vs [m]"])) and "not a field of the piece in" in fields_in["vs [m]: not defined"]


def test_a_direct_sum_of_flat_pieces(tmp_path):
    A, B = pg.L2(pg.Box((0, 1), (0, 1))), pg.L2(pg.Box((1, 2), (0, 1)))
    D = pg.DirectSum({"left": A, "right": B})
    x = D({"left": A.element(lambda x, y: x + y), "right": B.element(lambda x, y: x * y)}, name="x")
    rep = export(x, str(tmp_path / "ds.vtm"), nodes=11)
    _, pts, data, _ = _read(tmp_path / "ds", "right.vts")
    np.testing.assert_allclose(data["value"], pts[:, 0] * pts[:, 1], atol=1e-12)
    assert any("left: x = 1 | right: x = 1" == b for b in rep.boundaries)


@pytest.mark.skipif(PVBATCH is None, reason="ParaView (pvbatch) is not installed")
def test_paraview_reads_the_files_and_keeps_the_seam_sharp(tmp_path):
    export(_layered(), str(tmp_path / "m.vtm"), spacing=0.05)
    u = np.array([1.0, 0.3, 0.2]) / np.linalg.norm([1.0, 0.3, 0.2])
    script = tmp_path / "probe.py"
    script.write_text(textwrap.dedent(f"""
        from paraview.simple import XMLMultiBlockDataReader, ExtractBlock, ProbeLocation
        from paraview import servermanager as sm
        r = XMLMultiBlockDataReader(FileName=[{str(tmp_path / 'm.vtm')!r}])
        pieces = ExtractBlock(Input=r, Selectors=['/Root/pieces'])
        for rr in (0.549, 0.551):
            p = ProbeLocation(Input=pieces, ProbeType="Fixed Radius Point Source")
            p.ProbeType.Center = [rr * c for c in {[float(c) for c in u]!r}]
            p.UpdatePipeline()
            print("VALUE", sm.Fetch(p).GetPointData().GetArray("value").GetValue(0))
        """))
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    run = subprocess.run([PVBATCH, str(script)], capture_output=True, text=True, timeout=300, env=env)
    out = run.stdout
    assert "VALUE" in out, run.stderr[-2000:]
    got = [float(line.split()[1]) for line in out.splitlines() if line.startswith("VALUE")]
    exact = [IN(*(0.549 * u)), OUT(*(0.551 * u))]
    np.testing.assert_allclose(got, exact, atol=2e-3)
    assert abs(got[0] - got[1]) > 1.9, "the jump of 2 is kept between nodes 0.002 apart"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_heights_on_the_unit_sphere_are_drawn_at_their_radius(tmp_path):
    X, H = pg.L2(pg.Ball(2.0)), pg.L2(pg.Sphere(1.0))
    D = pg.DirectSum({"ball": X, "cmb": H})
    x = D({"ball": X.element(lambda x, y, z: x), "cmb": H.element(lambda x, y, z: z)}, name="x")
    rep = export(x, str(tmp_path / "h.vtm"), spacing=0.1, surfaces={"cmb": 1.2})
    _, pts, data, fields = _read(tmp_path / "h", "surface_cmb.vts")
    np.testing.assert_allclose(np.linalg.norm(pts, axis=1), 1.2, atol=1e-12)
    np.testing.assert_allclose(data["cmb"], pts[:, 2] / 1.2, atol=1e-6)           # the height at the unit point it came from
    assert [p["name"] for p in rep.pieces] == ["ball", "cmb (surface)"] and rep.surfaces == ["cmb"]
    assert max(s for s in rep.pieces[1]["steps"]) <= 0.1 + 1e-12
    a = rep.pieces[1]["arrays"][0]
    assert a["centres"] > 0 and np.isfinite(a["max"]) and a["max"] < 1e-2          # measured on the surface itself
    assert not os.path.exists(tmp_path / "h" / "cmb.vts")
