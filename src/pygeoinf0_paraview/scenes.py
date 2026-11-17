"""Ready ParaView scenes for an export (plan 056 P3): a script that ParaView runs to open the files with the controls of
pg.slices, all computed on the spot from the stored nodes (no frames are stored).

    path = scene(report, array=None, plane=(polar, azimuth, offset), radius=None, clim="robust", screenshot=None)
        writes <name>_scene.py beside the export: the pieces, a slice "plane" (normal from polar and azimuth in degrees,
        offset from the centre along it; its widget moves it), a slice "sphere" of the given radius about the centre
        (default: half the extent), the boundaries cut by the same plane (their rings follow the plane widget), a symmetric
        colour range ("robust": the 99th percentile of the array's nonzero |values|; "whole"; or (low, high), which with
        units applies to every member of the chosen field; other arrays get their own robust range), NaN transparent. The
        colour menu lists every (field, member) array: it is the member selector; every array has the same colour map and
        bar. array defaults to one with values in every piece. "volume (resampled)" (the pieces resampled to volume_dims^3,
        which volume rendering needs, so a jump is blurred over one voxel at seams; see-through, clear near zero) is in the
        pipeline, shown when volume=True. screenshot: a PNG path, saved when run.
    open_in_paraview(report, **scene options)    writes the scene and starts ParaView with it, if ParaView is found
    paraview()                                   the ParaView executable, or None

The script is plain pvpython: `paraview --script=<name>_scene.py` in the GUI, `pvbatch <name>_scene.py` without one. Opening
the .vtm itself in ParaView shows the data only (an outline), without the scene.
"""
import glob
import os
import shutil
import subprocess

TEMPLATE = '''"""Scene for {vtm} (pygeoinf0-paraview {version}). Open with  paraview --script={script_name}
Sampling and accuracy of every array: {txt}"""
import math
from paraview.simple import (ColorBy, ExtractBlock, GetActiveViewOrCreate, GetColorTransferFunction,
                             GetOpacityTransferFunction, GetScalarBar, Hide, Render, ResampleToImage, ResetCamera,
                             SaveScreenshot, SetActiveSource, Show, Slice, XMLMultiBlockDataReader)

S = {settings}
reader = XMLMultiBlockDataReader(registrationName="{name}", FileName=[S["vtm"]])
pieces = ExtractBlock(registrationName="pieces", Input=reader, Selectors=["/Root/pieces"])
view = GetActiveViewOrCreate("RenderView")
view.OrientationAxesVisibility = 1
pol, az, off = (math.radians(S["plane"][0]), math.radians(S["plane"][1]), S["plane"][2])
n = [math.sin(pol) * math.cos(az), math.sin(pol) * math.sin(az), math.cos(pol)]
c = S["centre"]

# fixed names: the widgets change the plane and the radius, so names holding their start values would go stale
plane = Slice(registrationName="plane", Input=pieces)
plane.SliceType = "Plane"
plane.SliceType.Normal = n
plane.SliceType.Origin = [c[k] + off * n[k] for k in range(3)]
pd = Show(plane, view)
ColorBy(pd, ("POINTS", S["array"]))

sphere = Slice(registrationName="sphere", Input=pieces)
sphere.SliceType = "Sphere"
sphere.SliceType.Center = c
sphere.SliceType.Radius = S["radius"]
sd = Show(sphere, view)
ColorBy(sd, ("POINTS", S["array"]))
sd.Opacity = S["sphere_opacity"]

if S["boundaries"]:
    walls = ExtractBlock(registrationName="boundaries", Input=reader, Selectors=["/Root/boundaries"])
    rings = Slice(registrationName="boundaries on the plane", Input=walls)
    rings.SliceType = plane.SliceType            # the same proxy: the rings follow the plane widget
    rd = Show(rings, view)
    rd.AmbientColor = rd.DiffuseColor = [0.0, 0.0, 0.0]
    rd.LineWidth = 2.0
    rd.ColorArrayName = ["POINTS", ""]

for name in S["surfaces"]:                       # boundary heights: in the pipeline, hidden (click the eye to show)
    ExtractBlock(registrationName="surface " + name, Input=reader, Selectors=["/Root/surfaces/" + name])

# a volume view: the pieces resampled to a regular grid, which volume rendering needs. It is an approximation of the export:
# each voxel takes the value of the piece its centre falls in, so a jump is blurred over one voxel and features narrower
# than a voxel are missed. Hidden unless volume=True; its name says so.
vol = ResampleToImage(registrationName="volume (resampled)", Input=pieces)
vol.UseInputBounds = 1
vol.SamplingDimensions = [S["volume_dims"]] * 3
vd = Show(vol, view)
vd.SetRepresentationType("Volume")
ColorBy(vd, ("POINTS", S["array"]))
vd.ScalarOpacityUnitDistance = 0.1 * S["extent"]   # see-through: the slices stay visible inside it
if not S["volume"]:
    Hide(vol, view)

# every array gets the same colour map and colour bar (with its own range), so another choice in the colour menu looks the
# same; a second array shown at the same time gets its bar beside the first instead of on top of it
for k, a in enumerate([S["array"]] + [a for a in S["arrays"] if a != S["array"]]):
    lo, hi = S["clims"][a]
    lut = GetColorTransferFunction(a)
    lut.ApplyPreset(S["cmap"], True)
    lut.RescaleTransferFunction(lo, hi)
    lut.AutomaticRescaleRangeMode = "Never"
    lut.NanOpacity = 0.0
    pwf = GetOpacityTransferFunction(a)            # for the volume: clear near zero, denser towards both ends of the range
    pwf.Points = [lo, 0.6, 0.5, 0.0, 0.5 * lo, 0.05, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5 * hi, 0.05, 0.5, 0.0, hi, 0.6, 0.5, 0.0]
    bar = GetScalarBar(lut, view)
    bar.Title = a + (" (" + S["units"][a] + ")" if S["units"].get(a) else "")
    bar.ComponentTitle = ""
    bar.WindowLocation = "Any Location"
    bar.Position = [0.9 - 0.2 * (k % 2), 0.1]      # a bar with its labels is about 0.2 of the view wide
    bar.Visibility = 0                              # ParaView shows the bar of an array when something is coloured by it
    bar.TextPosition = "Ticks left/bottom, annotations right/top"
    bar.TitleFontSize, bar.LabelFontSize = 7, 6
    bar.TitleFontFamily = bar.LabelFontFamily = "Times"    # ParaView's Arial draws [ ] as ( ), and the names hold [member]
    bar.HorizontalTitle = 1
    bar.ScalarBarThickness = 10
    bar.ScalarBarLength = 0.8
pd.SetScalarBarVisibility(view, True)

SetActiveSource(plane)
view.CameraFocalPoint = c
view.CameraPosition = [c[k] + 3.0 * S["extent"] * n[k] for k in range(3)]
view.CameraViewUp = [0.0, 0.0, 1.0] if abs(n[2]) < 0.9 else [0.0, 1.0, 0.0]
ResetCamera(view)
Render(view)
if S["screenshot"]:
    SaveScreenshot(S["screenshot"], view, ImageResolution=[1000, 800])
'''


def paraview():
    """The ParaView GUI executable: on the PATH, or in /Applications/ParaView*.app (macOS); None if absent."""
    found = shutil.which("paraview")
    if found:
        return found
    apps = sorted(glob.glob("/Applications/ParaView*.app/Contents/MacOS/paraview"))
    return apps[-1] if apps else None


def pvbatch():
    found = shutil.which("pvbatch")
    if found:
        return found
    apps = sorted(glob.glob("/Applications/ParaView*.app/Contents/bin/pvbatch"))
    return apps[-1] if apps else None


def scene(report, array=None, plane=(90.0, 0.0, 0.0), radius=None, clim="robust", cmap="Cool to Warm", screenshot=None,
          sphere_opacity=0.4, units=None, volume=False, volume_dims=128):
    """Write the scene script for an export's Report (module docstring); returns its path."""
    from . import __version__
    array = array or (report.volume_arrays or report.arrays)[0]
    field = array.split(" [")[0]                 # a given range and units apply to every member of the chosen field
    same = [a for a in report.arrays if a.split(" [")[0] == field]
    clims = {a: [float(v) for v in (clim if a in same and not isinstance(clim, str)
                                    else report.scale(a, clim if isinstance(clim, str) else "robust"))] for a in report.arrays}
    stem = os.path.splitext(report.path)[0]
    path = stem + "_scene.py"
    settings = dict(vtm=report.path, array=array, plane=[float(v) for v in plane], centre=report.centre, extent=report.extent,
                    radius=float(radius) if radius is not None else 0.5 * report.extent, cmap=cmap,
                    arrays=list(report.arrays), clims=clims,
                    boundaries=bool(report.boundaries), surfaces=list(getattr(report, "surfaces", [])), screenshot=os.path.abspath(screenshot) if screenshot else None,
                    sphere_opacity=float(sphere_opacity), units={a: units for a in same} if units else {}, volume=bool(volume), volume_dims=int(volume_dims))
    with open(path, "w") as f:
        f.write(TEMPLATE.format(vtm=os.path.basename(report.path), version=__version__, script_name=os.path.basename(path),
                                txt=os.path.basename(stem) + ".txt", name=os.path.basename(stem),
                                settings=repr(settings)))
    return path


def _env():
    return {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}


def open_in_paraview(report, **options):
    """Write the scene (scene(report, **options)) and start ParaView with it; without ParaView, say how to open it."""
    path = scene(report, **options)
    exe = paraview()
    if exe is None:
        print(f"ParaView was not found. Install it, then open the scene with: paraview --script={path}")
        return path
    subprocess.Popen([exe, f"--script={path}"], env=_env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return path
