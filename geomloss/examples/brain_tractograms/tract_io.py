"""
Input-Output with brain tractograms
==========================================

"""


import argparse
import os.path
import sys
import pdb
import numpy as np
from dipy.tracking.streamline import set_number_of_points

import vtk
from vtk.util import numpy_support as ns

try:
    # Python 2
    from itertools import izip
except ImportError:
    # Python 3
    izip = zip
try:
    # Python 2
    xrange
except NameError:
    # Python 3, xrange is now named range
    xrange = range


def read_vtk(filename):
    if filename.endswith("xml") or filename.endswith("vtp"):
        polydata_reader = vtk.vtkXMLPolyDataReader()
    else:
        polydata_reader = vtk.vtkPolyDataReader()

    polydata_reader.SetFileName(filename)
    polydata_reader.Update()

    polydata = polydata_reader.GetOutput()

    return vtkPolyData_to_tracts(polydata)


def vtkPolyData_to_tracts(polydata):
    result = {}
    result["lines"] = ns.vtk_to_numpy(polydata.GetLines().GetData())
    result["points"] = ns.vtk_to_numpy(polydata.GetPoints().GetData())
    result["numberOfLines"] = polydata.GetNumberOfLines()

    data = {}
    if polydata.GetPointData().GetScalars():
        data["ActiveScalars"] = polydata.GetPointData().GetScalars().GetName()
        result["Scalars"] = polydata.GetPointData().GetScalars()
    if polydata.GetPointData().GetVectors():
        data["ActiveVectors"] = polydata.GetPointData().GetVectors().GetName()
    if polydata.GetPointData().GetTensors():
        data["ActiveTensors"] = polydata.GetPointData().GetTensors().GetName()

    for i in xrange(polydata.GetPointData().GetNumberOfArrays()):
        array = polydata.GetPointData().GetArray(i)
        np_array = ns.vtk_to_numpy(array)
        if np_array.ndim == 1:
            np_array = np_array.reshape(len(np_array), 1)
        data[polydata.GetPointData().GetArrayName(i)] = np_array

    result["pointData"] = data

    tracts, data = vtkPolyData_dictionary_to_tracts_and_data(result)
    return tracts, data


def vtkPolyData_dictionary_to_tracts_and_data(dictionary):
    dictionary_keys = set(("lines", "points", "numberOfLines"))
    if not dictionary_keys.issubset(dictionary.keys()):
        raise ValueError(
            "Dictionary must have the keys lines and points" + repr(dictionary.keys())
        )

    tract_data = {}
    tracts = []

    lines = np.asarray(dictionary["lines"]).squeeze()
    points = dictionary["points"]

    actual_line_index = 0
    number_of_tracts = dictionary["numberOfLines"]
    original_lines = []
    for l in xrange(number_of_tracts):
        tracts.append(
            points[
                lines[
                    actual_line_index
                    + 1 : actual_line_index
                    + lines[actual_line_index]
                    + 1
                ]
            ]
        )
        original_lines.append(
            np.array(
                lines[
                    actual_line_index
                    + 1 : actual_line_index
                    + lines[actual_line_index]
                    + 1
                ],
                copy=True,
            )
        )
        actual_line_index += lines[actual_line_index] + 1

    if "pointData" in dictionary:
        point_data_keys = [
            it[0]
            for it in dictionary["pointData"].items()
            if isinstance(it[1], np.ndarray)
        ]

        for k in point_data_keys:
            array_data = dictionary["pointData"][k]
            if not k in tract_data:
                tract_data[k] = [array_data[f] for f in original_lines]
            else:
                np.vstack(tract_data[k])
                tract_data[k].extend(
                    [array_data[f] for f in original_lines[-number_of_tracts:]]
                )

    return tracts, tract_data


def save_vtk(filename, tracts, lines_indices=None, scalars=None):
    lengths = [len(p) for p in tracts]
    line_starts = ns.numpy.r_[0, ns.numpy.cumsum(lengths)]
    if lines_indices is None:
        lines_indices = [
            ns.numpy.arange(length) + line_start
            for length, line_start in izip(lengths, line_starts)
        ]

    ids = ns.numpy.hstack(
        [ns.numpy.r_[c[0], c[1]] for c in izip(lengths, lines_indices)]
    )
    vtk_ids = ns.numpy_to_vtkIdTypeArray(ids, deep=True)

    cell_array = vtk.vtkCellArray()
    cell_array.SetCells(len(tracts), vtk_ids)
    points = ns.numpy.vstack(tracts).astype(
        ns.get_vtk_to_numpy_typemap()[vtk.VTK_DOUBLE]
    )
    points_array = ns.numpy_to_vtk(points, deep=True)

    poly_data = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(points_array)
    poly_data.SetPoints(vtk_points)
    poly_data.SetLines(cell_array)
    poly_data.BuildCells()

    if filename.endswith(".xml") or filename.endswith(".vtp"):
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetDataModeToBinary()
    else:
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileTypeToBinary()

    writer.SetFileName(filename)
    if hasattr(vtk, "VTK_MAJOR_VERSION") and vtk.VTK_MAJOR_VERSION > 5:
        writer.SetInputData(poly_data)
    else:
        writer.SetInput(poly_data)
    writer.Write()


def save_vtk_labels(filename, tracts, scalars, lines_indices=None):
    lengths = [len(p) for p in tracts]
    line_starts = ns.numpy.r_[0, ns.numpy.cumsum(lengths)]
    if lines_indices is None:
        lines_indices = [
            ns.numpy.arange(length) + line_start
            for length, line_start in izip(lengths, line_starts)
        ]

    ids = ns.numpy.hstack(
        [ns.numpy.r_[c[0], c[1]] for c in izip(lengths, lines_indices)]
    )
    vtk_ids = ns.numpy_to_vtkIdTypeArray(ids, deep=True)

    cell_array = vtk.vtkCellArray()
    cell_array.SetCells(len(tracts), vtk_ids)
    points = ns.numpy.vstack(tracts).astype(
        ns.get_vtk_to_numpy_typemap()[vtk.VTK_DOUBLE]
    )
    points_array = ns.numpy_to_vtk(points, deep=True)

    poly_data = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(points_array)
    poly_data.SetPoints(vtk_points)
    poly_data.SetLines(cell_array)
    poly_data.GetPointData().SetScalars(ns.numpy_to_vtk(scalars))
    poly_data.BuildCells()
    #    poly_data.SetScalars(scalars)

    if filename.endswith(".xml") or filename.endswith(".vtp"):
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetDataModeToBinary()
    else:
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileTypeToBinary()

    writer.SetFileName(filename)
    if hasattr(vtk, "VTK_MAJOR_VERSION") and vtk.VTK_MAJOR_VERSION > 5:
        writer.SetInputData(poly_data)
    else:
        writer.SetInput(poly_data)
    writer.Write()


def streamlines_resample(streamlines, perc=None, npoints=None):
    if perc is not None:
        resampled = [
            set_number_of_points(s, int(len(s) * perc / 100.0)) for s in streamlines
        ]
    else:
        resampled = [set_number_of_points(s, npoints) for s in streamlines]

    return resampled


def check_ext(value):
    filename, file_extension = os.path.splitext(value)
    if file_extension in (".vtk", ".xml", ".vtp"):
        return value
    else:
        raise argparse.ArgumentTypeError(
            "Invalid file extension (file format supported: vtk,xml,vtp): %r" % value
        )


def check_resample(value):
    try:
        t = float(value)
        if 0 <= t < 100:
            return value
        else:
            raise argparse.ArgumentTypeError(
                "Invalid resampling (must be between 0 and 100): %r" % value
            )
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid resampling value: %r" % value)


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "Input_Tractogram", help="Name of the input tractography file", type=check_ext
    )
    parser.add_argument(
        "Percentage", help="Resampling percentage value", type=check_resample
    )

    args = parser.parse_args()

    return args.Input_Tractogram, float(args.Percentage)


if __name__ == "__main__":
    in_file, perc = setup()

    streamlines = read_vtk(in_file)[0]
    resampled = streamlines_resample(streamlines, perc)
    file_name, file_extension = os.path.splitext(in_file)
    print(file_name + "_resampled" + file_extension)
    save_vtk(file_name + "_resampled" + file_extension, np.array(resampled))

    sys.exit()


def save_tract(x, fname, NPOINTS=20):
    tract = x.view(len(x), -1, 3) * np.sqrt(NPOINTS)
    save_vtk(fname, tract.detach().cpu().numpy())


def save_tract_numpy(x, fname, NPOINTS=20):
    tract = x.view(len(x), -1, 3) * np.sqrt(NPOINTS)
    np.save(
        fname,
        np.float16(tract.detach().cpu().numpy()),
        allow_pickle=False,
        fix_imports=False,
    )


def save_tract_with_labels(fname, x, scalars, subsampling_fibers=None, NPOINTS=20):
    # save tracts +label information on each fiber
    tract = x.view(len(x), -1, 3) * np.sqrt(NPOINTS)
    tract = tract[0::subsampling_fibers, :, :]
    nf, ns, d = tract.shape
    labels = scalars.view(-1, 1).repeat(1, ns)
    save_vtk_labels(
        fname,
        tract.detach().cpu().numpy(),
        labels.view(-1).detach().cpu().numpy().astype(int),
    )


def save_tracts_labels_separate(fname, x, labels, start, end, NPOINTS=20):
    # save tracts +label information on each fiber
    tract = x.view(len(x), -1, 3) * np.sqrt(NPOINTS)
    for l in range(start, end):
        if (labels == l).nonzero().shape[0] != 0:
            tract_l = tract[(labels == l).nonzero().view(-1), :, :]
            save_vtk(fname + "_{:05d}.vtk".format(l), tract_l.detach().cpu().numpy())
        else:
            save_vtk(fname + "_{:05d}.vtk".format(l), np.array([[0, 0, 0]]))
