import numpy as np
from .base import BaseInterpolation
from scipy.sparse import csr_matrix


class GlsInterpolation(BaseInterpolation):
    def __init__(self, mesh):
        super().__init__(mesh)
        self.A = self._compute_diffusion_magnitude()
        self.in_vols_pairs = None
        self.Ns = None
        self.in_faces_map = None

    def interpolate(self):
        self._set_internal_vols_pairs()
        self._set_normal_vectors()
        self._set_in_faces_map()

        all_nodes = self.mesh.nodes.all[:]
        dirichlet_nodes_flag = self.mesh.dirichlet_nodes_flag[:].flatten()
        dirichlet_nodes = all_nodes[dirichlet_nodes_flag == 1]

        neumann_nodes_flag = self.mesh.neumann_nodes_flag[:].flatten()
        neumann_nodes = all_nodes[neumann_nodes_flag == 1]

        in_nodes = np.setdiff1d(
            np.setdiff1d(all_nodes, dirichlet_nodes, assume_unique=True),
            neumann_nodes, assume_unique=True)

        if len(in_nodes) > 0:
            ws, vols_idx = self._interpolate_internal_nodes(in_nodes)
            vols_idx_flat = np.concatenate(vols_idx)
            ws_flat = np.concatenate(ws)

            ns = np.array([len(vols) for vols in vols_idx])
            nodes_idx = np.repeat(in_nodes, ns)

            W_in = csr_matrix((ws_flat, (nodes_idx, vols_idx_flat)), shape=(
                len(all_nodes), len(self.mesh.volumes)))
        else:
            W_in = csr_matrix((len(all_nodes), len(self.mesh.volumes)))

        if len(neumann_nodes) > 0:
            ws, vols_idx, neumann_nodes_ws = self._interpolate_neumann_nodes(
                neumann_nodes)
            vols_idx_flat = np.concatenate(vols_idx)
            ws_flat = np.concatenate(ws)

            neu_ws = np.zeros(len(all_nodes))
            neu_ws[neumann_nodes] = neumann_nodes_ws[:]

            ns = np.array([len(vols) for vols in vols_idx])
            nodes_idx = np.repeat(neumann_nodes, ns)

            W_neu = csr_matrix((ws_flat, (nodes_idx, vols_idx_flat)), shape=(
                len(all_nodes), len(self.mesh.volumes)))
        else:
            W_neu = csr_matrix((len(all_nodes), len(self.mesh.volumes)))
            neu_ws = np.zeros(len(all_nodes))

        W = W_in + W_neu

        return W, neu_ws

    def _interpolate_internal_nodes(self, in_nodes):
        vols_around_in_nodes = self.mesh.nodes.bridge_adjacencies(
            in_nodes, 1, 3)
        faces_around_in_nodes = self.mesh.nodes.bridge_adjacencies(
            in_nodes, 1, 2)
        ws = []

        for v, KSetv, Sv in zip(in_nodes, vols_around_in_nodes, faces_around_in_nodes):
            nK = len(KSetv)
            nS = len(Sv)

            Mv = np.zeros((nK + 3 * nS, 3 * nK + 1))
            Nv = np.zeros((nK + 3 * nS, nK))

            self._assemble_ls_matrices(Mv, Nv, v, KSetv, Sv)

            M = np.linalg.inv(Mv.T @ Mv) @ (Mv.T @ Nv)
            w = M[-1, :]
            ws.append(w)

        return ws, vols_around_in_nodes

    def _interpolate_neumann_nodes(self, neumann_nodes):
        vols_around_neu_nodes = self.mesh.nodes.bridge_adjacencies(
            neumann_nodes, 1, 3)

        in_faces = self.mesh.faces.internal[:]
        neumann_faces_flag = self.mesh.neumann_faces[:].flatten()
        neumann_faces = self.mesh.faces.all[neumann_faces_flag == 1]
        gN = self.mesh.neumann[:].flatten()

        faces_around_neu_nodes = self.mesh.nodes.bridge_adjacencies(
            neumann_nodes, 1, 2)
        in_faces_around_neu_nodes = [
            np.intersect1d(f, in_faces, assume_unique=True)
            for f in faces_around_neu_nodes]
        neu_faces_around_neu_nodes = [
            np.intersect1d(f, neumann_faces, assume_unique=True)
            for f in faces_around_neu_nodes]

        ws = []
        neu_ws = []

        for v, KSetv, Sv, Svb in zip(
                neumann_nodes, vols_around_neu_nodes,
                in_faces_around_neu_nodes, neu_faces_around_neu_nodes):
            nK = len(KSetv)
            nS = len(Sv)
            nb = len(Svb)

            Mb = np.zeros((nK + 3 * nS + nb, 3 * nK + 1))
            Nb = np.zeros((nK + 3 * nS + nb, nK + 1))

            self._assemble_ls_matrices(Mb, Nb, v, KSetv, Sv)

            neu_rows = np.arange(start=nK + 3 * nS, stop=nK + 3 * nS + nb)

            Ks_Svb = self.mesh.faces.bridge_adjacencies(Svb, 2, 3).flatten()
            sorter = np.argsort(KSetv)
            Ik = sorter[np.searchsorted(KSetv, Ks_Svb, sorter=sorter)]

            N_svb = self.Ns[Svb]
            L = self.mesh.permeability[Ks_Svb].reshape((nb, 3, 3))
            nL = np.einsum("ij,ikj->ik", N_svb, L)

            Mb[neu_rows, 3 * Ik] = -nL[:, 0]
            Mb[neu_rows, 3 * Ik + 1] = -nL[:, 1]
            Mb[neu_rows, 3 * Ik + 2] = -nL[:, 2]

            Nb[neu_rows, nK] = gN[Svb]

            M = np.linalg.inv(Mb.T @ Mb) @ (Mb.T @ Nb)
            w = M[-1, :]

            wi = w[:-1]
            wc = w[-1]

            ws.append(wi)
            neu_ws.append(wc)

        return ws, vols_around_neu_nodes, neu_ws

    def _assemble_ls_matrices(self, Mi, Ni, v, KSetv, Sv):
        nK = len(KSetv)
        nS = len(Sv)

        xv = self.mesh.nodes.coords[v]
        xK = self.mesh.volumes.center[KSetv]
        dKv = xK - xv

        KSetV_range = np.arange(nK)
        Mi[KSetV_range, 3 * KSetV_range] = dKv[:, 0]
        Mi[KSetV_range, 3 * KSetV_range + 1] = dKv[:, 1]
        Mi[KSetV_range, 3 * KSetV_range + 2] = dKv[:, 2]
        Mi[KSetV_range, 3 * nK] = 1.0

        Ni[KSetV_range, KSetV_range] = 1.0

        Sv_in_idx = self.in_faces_map[Sv]
        Sv_in_idx = Sv_in_idx[Sv_in_idx != -1]
        Ks_Sv = self.in_vols_pairs[Sv_in_idx, :]
        sorter = np.argsort(KSetv)
        Ij1 = sorter[np.searchsorted(KSetv, Ks_Sv[:, 0], sorter=sorter)]
        Ij2 = sorter[np.searchsorted(KSetv, Ks_Sv[:, 1], sorter=sorter)]

        xS = self.mesh.faces.center[Sv]
        eta_j = np.max(self.A[Ks_Sv], axis=1)
        N_sj = self.Ns[Sv]
        T_sj1 = xv - xS
        T_sj2 = np.cross(N_sj, T_sj1)
        tau_j2 = np.linalg.norm(T_sj2, axis=1) ** (-eta_j)
        tau_tsj2 = tau_j2[:, np.newaxis] * T_sj2

        L1 = self.mesh.permeability[Ks_Sv[:, 0]].reshape((nS, 3, 3))
        L2 = self.mesh.permeability[Ks_Sv[:, 1]].reshape((nS, 3, 3))
        nL1 = np.einsum("ij,ikj->ik", N_sj, L1)
        nL2 = np.einsum("ij,ikj->ik", N_sj, L2)

        idx1 = np.arange(start=nK, stop=nK + 3 * nS - 2, step=3)
        idx2 = np.arange(start=nK + 1, stop=nK + 3 * nS - 1, step=3)
        idx3 = np.arange(start=nK + 2, stop=nK + 3 * nS, step=3)

        Mi[idx1, 3 * Ij1] = -nL1[:, 0]
        Mi[idx1, 3 * Ij1 + 1] = -nL1[:, 1]
        Mi[idx1, 3 * Ij1 + 2] = -nL1[:, 2]

        Mi[idx1, 3 * Ij2] = nL2[:, 0]
        Mi[idx1, 3 * Ij2 + 1] = nL2[:, 1]
        Mi[idx1, 3 * Ij2 + 2] = nL2[:, 2]

        Mi[idx2, 3 * Ij1] = -T_sj1[:, 0]
        Mi[idx2, 3 * Ij1 + 1] = -T_sj1[:, 1]
        Mi[idx2, 3 * Ij1 + 2] = -T_sj1[:, 2]

        Mi[idx2, 3 * Ij2] = T_sj1[:, 0]
        Mi[idx2, 3 * Ij2 + 1] = T_sj1[:, 1]
        Mi[idx2, 3 * Ij2 + 2] = T_sj1[:, 2]

        Mi[idx3, 3 * Ij1] = -tau_tsj2[:, 0]
        Mi[idx3, 3 * Ij1 + 1] = -tau_tsj2[:, 1]
        Mi[idx3, 3 * Ij1 + 2] = -tau_tsj2[:, 2]

        Mi[idx3, 3 * Ij2] = tau_tsj2[:, 0]
        Mi[idx3, 3 * Ij2 + 1] = tau_tsj2[:, 1]
        Mi[idx3, 3 * Ij2 + 2] = tau_tsj2[:, 2]

    def _set_internal_vols_pairs(self):
        """Set the pairs of volumes sharing an internal face in the 
        attribute `in_vols_pairs`.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        internal_faces = self.mesh.faces.internal[:]
        self.in_vols_pairs = self.mesh.faces.bridge_adjacencies(
            internal_faces, 2, 3)
    
    def _set_in_faces_map(self):
        internal_faces = self.mesh.faces.internal[:]
        self.in_faces_map = np.zeros(len(self.mesh.faces), dtype=int)
        self.in_faces_map[:] = -1
        self.in_faces_map[internal_faces] = np.arange(internal_faces.shape[0])

    def _set_normal_vectors(self):
        """Compute and store the normal vectors and their norms.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Retrieve the internal faces.
        all_faces = self.mesh.faces.all[:]

        # Retrieve the points that form the components of the normal vectors.
        faces_nodes = self.mesh.faces.bridge_adjacencies(all_faces, 0, 0)
        I_idx = faces_nodes[:, 0]
        J_idx = faces_nodes[:, 1]
        K_idx = faces_nodes[:, 2]

        I = self.mesh.nodes.coords[I_idx]
        J = self.mesh.nodes.coords[J_idx]
        K = self.mesh.nodes.coords[K_idx]

        # Set the normal vectors.
        Ns = np.cross(I - J, K - J)
        Ns_norm = np.linalg.norm(Ns, axis=1)
        self.Ns = Ns / Ns_norm[:, np.newaxis]

    def _compute_diffusion_magnitude(self):
        nvols = len(self.mesh.volumes)
        Ks = self.mesh.permeability[:].reshape((nvols, 3, 3))

        detKs = np.linalg.det(Ks)
        trKs = np.trace(Ks, axis1=1, axis2=2)

        A = (1 - (3 * (detKs ** (1 / 3)) / trKs)) ** 2

        return A
