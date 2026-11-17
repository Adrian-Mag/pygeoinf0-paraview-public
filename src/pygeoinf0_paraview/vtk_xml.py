"""VTK XML files with numpy only: structured grids (.vts), multiblock (.vtm). Arrays are appended raw, little endian, with UInt64
size headers, so no vtk module is needed to write and the values are stored in full double precision."""
import os
from xml.sax.saxutils import quoteattr

import numpy as np

TYPES = {np.dtype("float64"): "Float64", np.dtype("float32"): "Float32", np.dtype("int32"): "Int32", np.dtype("uint8"): "UInt8"}


def _string_array(name, text):
    codes = " ".join(str(b) for b in text.encode("utf-8")) + " 0"
    return f'<Array type="String" Name={quoteattr(name)} NumberOfTuples="1" format="ascii">{codes}</Array>\n'


def write_vts(path, shape, points, point_data, field_data=None):
    """A structured grid of shape (n0, n1, n2) (index 0 fastest, as VTK), points (n0 n1 n2, 3) in that order, point_data
    {name: (n0 n1 n2,) or (.., k)}, field_data {name: text}."""
    shape = tuple(int(s) for s in shape) + (1,) * (3 - len(shape))
    ext = " ".join(f"0 {s - 1}" for s in shape)
    blobs, heads, offset = [], [], 0

    def add(name, a, comps):
        nonlocal offset
        a = np.ascontiguousarray(a, dtype=np.float64 if a.dtype.kind == "f" else a.dtype)
        if a.dtype.byteorder == ">":
            a = a.byteswap().view(a.dtype.newbyteorder("<"))
        raw = a.tobytes()
        heads.append(f'<DataArray type="{TYPES[np.dtype(a.dtype.name)]}" Name={quoteattr(name)} NumberOfComponents="{comps}" '
                     f'format="appended" offset="{offset}"/>\n')
        blobs.append(np.uint64(len(raw)).tobytes() + raw)
        offset += 8 + len(raw)

    pd = []
    for name, a in point_data.items():
        a = np.asarray(a)
        add(name, a, 1 if a.ndim == 1 else a.shape[1])
        pd.append(heads.pop())
    add("Points", np.asarray(points, float).reshape(-1, 3), 3)
    pts = heads.pop()
    with open(path, "wb") as f:
        f.write(b'<?xml version="1.0"?>\n<VTKFile type="StructuredGrid" version="1.0" byte_order="LittleEndian" '
                b'header_type="UInt64">\n')
        f.write(f'<StructuredGrid WholeExtent="{ext}">\n'.encode())
        if field_data:
            f.write(b"<FieldData>\n" + "".join(_string_array(k, v) for k, v in field_data.items()).encode() + b"</FieldData>\n")
        first = next(iter(point_data), None)
        f.write(f'<Piece Extent="{ext}">\n<PointData{" Scalars=" + quoteattr(first) if first else ""}>\n'.encode())
        f.write("".join(pd).encode() + b"</PointData>\n<Points>\n" + pts.encode() + b"</Points>\n</Piece>\n</StructuredGrid>\n")
        f.write(b'<AppendedData encoding="raw">\n_' + b"".join(blobs) + b"\n</AppendedData>\n</VTKFile>\n")


def write_vtm(path, blocks):
    """blocks: [(name, file relative to path's folder)] or [(name, [(name, file), ...])] for nested groups."""
    def body(items, depth):
        out = []
        for i, (name, item) in enumerate(items):
            if isinstance(item, list):
                out.append(f'{"  " * depth}<Block index="{i}" name={quoteattr(name)}>\n' + body(item, depth + 1)
                           + f'{"  " * depth}</Block>\n')
            else:
                out.append(f'{"  " * depth}<DataSet index="{i}" name={quoteattr(name)} file={quoteattr(item)}/>\n')
        return "".join(out)
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="vtkMultiBlockDataSet" version="1.0">\n<vtkMultiBlockDataSet>\n'
                + body(blocks, 1) + "</vtkMultiBlockDataSet>\n</VTKFile>\n")


def read_vts(path):
    """(shape, points, point_data, field_data) of a file written by write_vts (for tests; no vtk needed)."""
    import re
    raw = open(path, "rb").read()
    head, data = raw.split(b'<AppendedData encoding="raw">\n_', 1)
    text = head.decode()
    ext = [int(v) for v in re.search(r'WholeExtent="([^"]+)"', text).group(1).split()]
    shape = tuple(ext[2 * k + 1] + 1 for k in range(3))
    arrays = {}
    for m in re.finditer(r'<DataArray type="(\w+)" Name="([^"]+)" NumberOfComponents="(\d+)" format="appended" '
                         r'offset="(\d+)"/>', text):
        typ, name, comps, off = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        n = int(np.frombuffer(data[off:off + 8], "<u8")[0])
        a = np.frombuffer(data[off + 8: off + 8 + n], {"Float64": "<f8", "Float32": "<f4", "Int32": "<i4", "UInt8": "u1"}[typ])
        arrays[name] = a.reshape(-1, comps) if comps > 1 else a
    fields = {m.group(1): bytes(int(c) for c in m.group(2).split()[:-1]).decode()
              for m in re.finditer(r'<Array type="String" Name="([^"]+)" NumberOfTuples="1" format="ascii">([^<]*)</Array>', text)}
    pts = arrays.pop("Points")
    return shape, pts, arrays, fields


def relpath(path, start):
    return os.path.relpath(path, os.path.dirname(os.path.abspath(start)))
