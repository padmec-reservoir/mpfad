import numpy as np
from .base import BaseInterpolation
from scipy.sparse import csr_matrix


class IdwInterpolation(BaseInterpolation):
    def interpolate(self):
        all_nodes = self.mesh.nodes.all[:]
        dirichlet_nodes_flag = self.mesh.dirichlet_nodes_flag[:].flatten()
        dirichlet_nodes = all_nodes[dirichlet_nodes_flag == 1]

        vols_around = self.mesh.nodes.bridge_adjacencies(all_nodes, 1, 3)
        vols_around_flat = np.concatenate(vols_around)
        C = self.mesh.volumes.center[vols_around_flat]

        ns = np.array([len(vols) for vols in vols_around])
        nodes_idx = np.repeat(all_nodes, ns)
        P = self.mesh.nodes.coords[nodes_idx]

        D = np.linalg.norm(P - C, axis=1)
        D_inv = 1 / D

        for node in all_nodes:
            prev = np.sum(ns[: node])
            curr = np.sum(ns[: (node + 1)])

            if node in dirichlet_nodes:
                D_inv[prev:curr] = 0
            else:
                D_inv[prev:curr] /= np.sum(D_inv[prev:curr])

        W = csr_matrix((D_inv, (nodes_idx, vols_around_flat)),
                       shape=(len(all_nodes), len(self.mesh.volumes)))

        neu_ws = np.zeros(len(all_nodes))

        return W, neu_ws
