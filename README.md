# pygeoinf0-paraview

A pygeoinf0 plugin (plan 056): functions on pygeoinf0's domains, split domains (pg.Layered, pg.OnPieces) and direct sums as
VTK files, with ready ParaView scenes. Writing needs numpy only; ParaView, installed by the user, opens the files. Saved
views reopen without Python, and every slice is cut on the spot from the stored nodes: no frames are stored.

```python
import pygeoinf0_paraview as pv
report = pv.export(model, "model.vtm", spacing=0.02, units={"vs": "km/s"})   # f, a family or a list of elements
print(report)                                       # what was written and what it cannot show
pv.open_in_paraview(report, plane=(60, 30, 0.1), radius=0.8)                  # or pv.scene(...): the script only
```

- **One block per piece.** Each piece is gridded on its own chart, with nodes on all its boundaries, so a jump at a seam
  stays a jump; a seam is written once under "boundaries", and the scene cuts it with the plane (rings that follow the widget).
- **Boundary heights** (summands on the unit sphere outside the partition, as in the normal-mode model space):
  `surfaces={"cmb": 3480.0}` draws them on the sphere of that radius under "surfaces" (hidden in the scene: click the eye).
- **Fields missing on a piece** (vs in the outer core) are written there as NaN, transparent, so every array exists in every
  block and filters such as ResampleToImage keep it; the field data says so. The scene's default array has values in every piece.
- **Volume rendering** needs a regular grid: the scene has "volume (resampled)" (the pieces resampled to 128^3, so a jump is
  blurred over one voxel at seams; see-through, clear near zero), shown with `volume=True` or by clicking its eye.
- **Open the scene, not the .vtm:** `paraview --script=model_scene.py` (or `open_in_paraview`). Opening the .vtm shows only the
  data.
- **In the scene:** select "plane" or "sphere" in the Pipeline Browser to show its widget; after dragging it, or typing a
  radius, click **Apply** (Properties panel): the cut and the boundary rings move only then. The names stay "plane" and
  "sphere" whatever their settings; the start values are in the scene script's `S`.
- **Members and fields** are arrays named `field` or `field [member]`; ParaView's colour menu selects them. Every array has the
  same colour map and colour bar, with a symmetric range of its own (a `clim=` given applies to every member of the chosen
  field); two arrays shown at once get two bars side by side. The menu also lists the text arrays `piece`, `sampling`,
  `field [member]: accuracy` and `…: interpolation` (the error account, below) and ParaView's own `vtkCompositeIndex` and
  `vtkBlockColors`: colouring by them shows nothing useful.
- **The error account per block** is in its field data. ParaView's View menu has no Spreadsheet entry: split the view (the
  button at the view's top right), choose SpreadSheet View (scroll to the bottom of the list), then in the spreadsheet's
  toolbar set Showing to "pieces" and Attribute to Field Data. The text is wider than the pane: scroll right, or read the
  same text in `model.txt`. While the spreadsheet is the active view, the colour and representation menus are greyed out:
  they act on the active view.
- **Apply is lit when the scene opens**, for every item: ParaView marks what a script made as not yet applied. The scene is
  complete as shown; in the GUI checks, clicking Apply after moving a widget changed nothing else.
- **Error account** (the package's rule: mathematical truth first). Per piece and array, the report (in `model.txt` and in each
  block's field data) states the nodes and largest steps, the accuracy of the node values (formulas: exact; realisations: their
  estimated error in their space's norm, which bounds no single value), ParaView's interpolation error measured at every cell
  centre against the exact part (largest and RMS: measured, not bounded between centres), and the faceting of curved cells. A
  feature narrower than about two steps can lie between nodes.
- **A portable page:** `pv.html(model, "model.html", spacing=...)` writes one self-contained HTML file that cuts planes and
  spheres (as latitude-longitude maps) on the spot from each piece's own nodes, so seams stay sharp; it states its float32
  storage and its own measured interpolation error. Balls, shells and 3-D boxes; other domains are refused by name.
- **Without ParaView** the files and the scene are still written, and `open_in_paraview` says how to open them.

Upstream github.com/Adrian-Mag/pygeoinf0-paraview-public. Status: 0.1.0, port 1 (needs `pg.port.fields`, pygeoinf0 with plan 056 P1). Tests: `pytest tests` (17; four run ParaView's
pvbatch and two the page's JavaScript in Node.js when installed, and are skipped otherwise). Not yet: measures and kernels drawn as points, rays or cell data;
cubed-sphere grids (the (r, polar, azimuth) chart crowds its nodes
at the poles and the centre). Released under the BSD 3-Clause licence (LICENSE). Install pygeoinf0 from
github.com/Adrian-Mag/pygeoinf0-public first. Mentions of plans refer to the author's development records, which are not
included.
