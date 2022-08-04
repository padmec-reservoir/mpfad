from utils import run_example
from preprocessor.meshHandle.finescaleMesh import FineScaleMesh
import numpy as np
import sys


def assign_mesh_properties(mesh):
    r1 = mesh.volumes.flag[1]
    r2 = mesh.volumes.flag[2]
    r3 = mesh.volumes.flag[3]

    K = np.zeros((len(mesh.volumes), 9))

    K[r1] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    K[r2] = np.array([0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 1.0])
    K[r3] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    mesh.permeability[:] = K[:]

    bfaces = mesh.faces.boundary[:]
    bfaces_centers = mesh.faces.center[bfaces]
    dirichlet_faces = bfaces[(bfaces_centers[:, 0] == 0.0) |
                             (bfaces_centers[:, 0] == 1.0) |
                             (bfaces_centers[:, 1] == 0.0) |
                             (bfaces_centers[:, 1] == 1.0)]
    mesh.dirichlet_faces[dirichlet_faces] = 1

    bnodes = mesh.nodes.boundary[:]

    dirichlet_nodes = mesh.faces.connectivities[dirichlet_faces].flatten()
    dnodes_coords = mesh.nodes.coords[dirichlet_nodes]
    mesh.dirichlet_nodes[dirichlet_nodes] = solution(
        dnodes_coords[:, 0], dnodes_coords[:, 1])
    mesh.dirichlet_nodes_flag[dirichlet_nodes] = 1

    neumann_nodes = bnodes[~np.isin(bnodes, dirichlet_nodes)]
    neumann_faces_idx = bfaces[~np.isin(bfaces, dirichlet_faces)]
    mesh.neumann_faces[neumann_faces_idx] = 1
    mesh.neumann_nodes_flag[neumann_nodes] = 1


def solution(x, y, z=None):
    delta = 0.2
    return - x - delta * y


def main():
    mesh_file = "../data/oblique-fracture_1.msh"
    interpolation_method = sys.argv[1]
    export_results = sys.argv[2]

    run_example(mesh_file, interpolation_method, export_results,
                assign_mesh_properties, solution)


if __name__ == "__main__":
    main()
