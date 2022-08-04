from typing import Any, Callable
from mpfad.core.schemes import MpfadScheme
from mpfad.interpolations.gls import GlsInterpolation
from mpfad.interpolations.idw import IdwInterpolation
from preprocessor.meshHandle.finescaleMesh import FineScaleMesh
from scipy.sparse.linalg import spsolve
import numpy as np


def export_mesh_file(mesh, fname: str):
    meshset = mesh.core.mb.create_meshset()
    mesh.core.mb.add_entities(meshset, mesh.core.all_volumes)
    mesh.core.mb.write_file(fname, [meshset])


def run_example(
        mesh_file: str, interpolation_method: str, export_results: str,
        setup_func: Callable[[Any],
                             None],
        solution:
        Callable
        [['np.ndarray[np.bool]', 'np.ndarray[np.bool]', 'np.ndarray[np.bool]'],
         'np.ndarray[np.bool]']) -> None:
    """Runs a MPFA-D example.

    Args:
        mesh_file (str):
            Mesh file name.
        interpolation_method (string):
            Interpolation method for the node variables in the MPFA-D.
        export_results (string):
            Flag to indicate if a mesh file with the pressure field must be exported.
        setup_func (Callable):
            Function that assigns the required mesh properties to the mesh object.
        solution (Callable):
            Function that computes the exact solution of the problem.
    """
    mesh = FineScaleMesh(mesh_file)

    setup_func(mesh)

    if interpolation_method == "gls":
        interpolation = GlsInterpolation(mesh)
    elif interpolation_method == "idw":
        interpolation = IdwInterpolation(mesh)
    else:
        raise ValueError(
            'Invalid interpolation method. Must be one of "gls" or "idw".')

    mpfad_instance = MpfadScheme(mesh, interpolation)

    A, q = mpfad_instance.assemble()

    p = spsolve(A, q)

    C = mesh.volumes.center[:]
    p_exact = solution(C[:, 0], C[:, 1], C[:, 2])

    V = mesh.volumes.volume[:]
    l2_rel_u = (np.sum(((p_exact - p) ** 2)
                * V) / np.sum((p_exact ** 2) * V)) ** 0.5

    print("Pressure L2 error: {}".format(l2_rel_u))

    if export_results == "y":
        fname = "./results/%s_result.vtk" % mesh_file.split("/")[-1].split(".")[0]
        mesh.pressure[:] = p[:]
        export_mesh_file(mesh, fname)
