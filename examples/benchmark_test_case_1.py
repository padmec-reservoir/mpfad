from utils import run_example
import numpy as np
import sys


def assign_mesh_properties(mesh):
    mesh.permeability[:] = np.array(
        [1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0])

    dirichlet_faces = mesh.faces.boundary[:]
    mesh.dirichlet_faces[dirichlet_faces] = 1
    dirichlet_nodes = mesh.faces.connectivities[dirichlet_faces].flatten()
    ns = mesh.nodes.coords[dirichlet_nodes]
    mesh.dirichlet_nodes[dirichlet_nodes] = solution(
        ns[:, 0], ns[:, 1], ns[:, 2])
    mesh.dirichlet_nodes_flag[dirichlet_nodes] = 1

    C = mesh.volumes.center[:]
    mesh.source_term[:] = source(mesh, C[:, 0], C[:, 1], C[:, 2])


def solution(x, y, z):
    return 1 + np.sin(np.pi * x) * np.sin(np.pi * (y + 0.5)) * np.sin(np.pi *
                                                                      (z + (1 / 3)))


def source(mesh, x, y, z):
    sx, sy, sz = np.sin(np.pi * x), np.sin(np.pi * (y + 1/2)
                                           ), np.sin(np.pi * (z + 1/3))
    cx, cy, cz = np.cos(np.pi * x), np.cos(np.pi * (y + 1/2)
                                           ), np.cos(np.pi * (z + 1/3))

    q = (np.pi ** 2) * (3 * sx * sy * sz - sx * cy * cz - sz * cx * cy)
    V = mesh.volumes.volume[:]

    Q = q * V

    return Q


def main():
    mesh_file = "../data/benchtetra2.msh"
    interpolation_method = sys.argv[1]
    export_results = sys.argv[2]

    run_example(mesh_file, interpolation_method, export_results,
                assign_mesh_properties, solution)


if __name__ == "__main__":
    main()
