from utils import run_example
from sympy.vector import gradient, CoordSys3D, divergence
from sympy.utilities.lambdify import lambdify
import sympy as sp
import numpy as np
import sys


def assign_mesh_properties(mesh):
    C = mesh.volumes.center[:]
    mesh.permeability[:] = K(C[:, 0], C[:, 1], C[:, 2])

    dirichlet_faces = mesh.faces.boundary[:]
    mesh.dirichlet_faces[dirichlet_faces] = 1
    dirichlet_nodes = mesh.faces.connectivities[dirichlet_faces].flatten()
    ns = mesh.nodes.coords[dirichlet_nodes]
    mesh.dirichlet_nodes[dirichlet_nodes] = solution(
        ns[:, 0], ns[:, 1], ns[:, 2])
    mesh.dirichlet_nodes_flag[dirichlet_nodes] = 1

    mesh.source_term[:] = source(mesh)


def K(x, y, z):
    K = np.zeros((len(x), 9))

    a = (x ** 2 + y ** 2) ** -1
    eps_x = 1
    eps_y = 1e-3
    eps_z = 10

    K[:, 0] = a * (eps_x * (x ** 2) + eps_y * (y ** 2))
    K[:, 1] = K[:, 3] = a * (eps_x - eps_y) * x * y
    K[:, 4] = a * (eps_y * (x ** 2) + eps_x * (y ** 2))
    K[:, 8] = eps_z * (z + 1)

    return K


def solution(x, y, z):
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.sin(2 * np.pi * z)


def source(mesh):
    eps_x = 1
    eps_y = 1e-3
    eps_z = 10

    R = CoordSys3D("R")
    a = (R.x ** 2 + R.y ** 2) ** -1

    K = sp.ImmutableMatrix(
        [[a * (eps_x * (R.x ** 2) + eps_y * (R.y ** 2)),
          (eps_x - eps_y) * a * R.x * R.y, 0.0],
         [(eps_x - eps_y) * a * R.x * R.y, a *
          (eps_y * R.x ** 2 + eps_x * R.y ** 2),
          0.0],
         [0.0, 0.0, eps_z * a * (R.z + 1) * (R.x ** 2 + R.y ** 2)]])
    u = sp.sin(2 * sp.pi * R.x) * sp.sin(2 * sp.pi * R.y) * sp.sin(2 * sp.pi * R.z)

    grad_u = gradient(u)
    K_grad_u = (-K) @ grad_u.to_matrix(R)
    vec_K_grad_u = K_grad_u[0] * R.i + K_grad_u[1] * R.j + K_grad_u[2] * R.k
    div_K_grad_u = divergence(vec_K_grad_u)

    div_K_grad_u_func = lambdify([R.x, R.y, R.z], div_K_grad_u)

    C = mesh.volumes.center[:]
    V = mesh.volumes.volume[:]

    q = np.zeros(len(mesh.volumes))

    for i, c in enumerate(C):
        q[i] = div_K_grad_u_func(c[0], c[1], c[2])

    Q = q * V

    return Q


def main():
    mesh_file = "../data/benchtetra1.msh"
    interpolation_method = sys.argv[1]
    export_results = sys.argv[2]

    run_example(mesh_file, interpolation_method, export_results,
                assign_mesh_properties, solution)


if __name__ == "__main__":
    main()
