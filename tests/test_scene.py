"""pygeoinf0-paraview scenes (plan 056 P3): the generated script runs under pvbatch and saves a picture; without ParaView the
scene is still written and the user is told how to open it."""
import json
import os
import subprocess

import numpy as np
import pytest

import pygeoinf0 as pg
import pygeoinf0_paraview as pv
from pygeoinf0_paraview import scenes as S


def _export(tmp_path):
    P = pg.LayeredBall([0.55, 1.0], names=["core", "mantle"])
    V = pg.PiecewiseSpace.lebesgue(P)
    x = V({"core": lambda x, y, z: 2 + x, "mantle": lambda x, y, z: 0.5 * y}, name="m")
    return pv.export([x, -x], str(tmp_path / "m.vtm"), spacing=0.1)


def test_the_colour_range_is_symmetric_and_robust(tmp_path):
    rep = _export(tmp_path)
    lo, hi = rep.scale("value [m]")
    assert lo == -hi and 2.0 < hi <= 2.55
    assert rep.scale("value [m]", "whole")[1] == pytest.approx(2.55, abs=1e-9)
    with pytest.raises(KeyError):
        rep.scale("nothing")


def test_without_paraview_the_scene_is_written_and_explained(tmp_path, monkeypatch, capsys):
    rep = _export(tmp_path)
    monkeypatch.setattr(S, "paraview", lambda: None)
    path = pv.open_in_paraview(rep, plane=(60, 30, 0.1))
    assert os.path.exists(path) and "paraview --script=" in capsys.readouterr().out
    text = open(path).read()
    assert "'plane': [60.0, 30.0, 0.1]" in text and "rings.SliceType = plane.SliceType" in text


@pytest.mark.skipif(S.pvbatch() is None, reason="ParaView (pvbatch) is not installed")
def test_the_scene_runs_headless_and_saves_a_picture(tmp_path):
    rep = _export(tmp_path)
    png = tmp_path / "m.png"
    path = pv.scene(rep, array="value [-m]" if "value [-m]" in rep.arrays else rep.arrays[1], radius=0.8, screenshot=str(png))
    run = subprocess.run([S.pvbatch(), path], env=S._env(), capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    assert png.exists() and png.stat().st_size > 10000


@pytest.mark.skipif(S.pvbatch() is None, reason="ParaView (pvbatch) is not installed")
def test_the_volume_view_keeps_a_field_missing_on_some_pieces(tmp_path):
    part = pg.Partition(pg.Ball(1.0), [pg.Ball(0.55), pg.Shell(0.55, 1.0)], names=["in", "out"])
    M = pg.PiecewiseSpace(part, {"vp": pg.L2, "vs": pg.L2}, where={"vs": ["out"]})
    rep = pv.export(M.split({"vp": lambda x, y, z: 1.0 + x, "vs": lambda x, y, z: 2.0 + y}), str(tmp_path / "v.vtm"), spacing=0.15)
    assert rep.volume_arrays == ["vp"]
    path = pv.scene(rep, array="vs", volume=True, volume_dims=32, screenshot=str(tmp_path / "v.png"))
    probe = open(path).read() + """
from paraview import servermanager as sm
img = sm.Fetch(vol)
print("ARRAYS", sorted(img.GetPointData().GetArrayName(i) for i in range(img.GetPointData().GetNumberOfArrays())))
"""
    open(path, "w").write(probe)
    run = subprocess.run([S.pvbatch(), path], env=S._env(), capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    line = next(l for l in run.stdout.splitlines() if l.startswith("ARRAYS"))
    assert "'vs'" in line and "'vp'" in line


def test_every_array_gets_a_colour_map_and_a_given_range_covers_the_field(tmp_path):
    rep = _export(tmp_path)
    path = pv.scene(rep, array="value [m]", clim=(-3, 3), units="km/s")
    text = open(path).read()
    S_ = eval(text.split("S = ", 1)[1].split("\n", 1)[0])
    assert set(S_["clims"]) == set(rep.arrays) and all(S_["clims"][a] == [-3.0, 3.0] for a in rep.arrays)
    assert set(S_["units"]) == set(rep.arrays)                      # both members are the field "value"
    assert 'registrationName="plane"' in text and 'registrationName="sphere"' in text


@pytest.mark.skipif(S.pvbatch() is None, reason="ParaView (pvbatch) is not installed")
def test_the_pipeline_names_stay_fixed_and_every_array_has_the_same_colour_map(tmp_path):
    rep = _export(tmp_path)
    other = rep.arrays[1]
    path = pv.scene(rep, array=rep.arrays[0], radius=0.8)
    probe = open(path).read() + f"""
import json
from paraview.simple import GetSources
sd.SMProxy.SetScalarColoring({other!r}, 0)       # what ParaView's colour menu calls: the scene's range must stay
out = dict(names=sorted(k[0] for k in GetSources()), luts={{}})
for a in S["arrays"]:
    l = GetColorTransferFunction(a)
    out["luts"][a] = dict(low=list(l.RGBPoints)[1:4], range=[l.RGBPoints[0], l.RGBPoints[-4]],
                          font=[GetScalarBar(l, view).TitleFontSize, str(GetScalarBar(l, view).TitleFontFamily).strip("'")])
print("OUT", json.dumps(out))
"""
    open(path, "w").write(probe)
    run = subprocess.run([S.pvbatch(), path], env=S._env(), capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    out = json.loads(next(l for l in run.stdout.splitlines() if l.startswith("OUT"))[4:])
    for n in ("plane", "sphere", "volume (resampled)", "boundaries on the plane"):
        assert n in out["names"]
    luts = out["luts"]
    assert set(luts) == set(rep.arrays) and len({(tuple(v["low"]), tuple(v["font"])) for v in luts.values()}) == 1
    assert luts[other]["range"] == pytest.approx(list(rep.scale(other)))
    assert all(v["font"][1] == "Times" for v in luts.values()), luts   # ParaView's Arial draws the [ ] of "field [member]" as ( )
