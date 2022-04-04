import numpy as np
from .base import BaseInterpolation
from scipy.sparse import lil_matrix


class IdwInterpolation(BaseInterpolation):
    def interpolate(self):
        all_nodes = self.mesh.nodes.all[:]

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
            D_inv[prev:curr] /= np.sum(D_inv[prev:curr])

        W = lil_matrix((len(all_nodes), len(self.mesh.volumes)))
        W[nodes_idx, vols_around_flat] = D_inv[:]

        return W.tocsr()
