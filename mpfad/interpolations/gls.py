import numpy as np
from .base import BaseInterpolation
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv


class GlsInterpolation(BaseInterpolation):
    def __init__(self, mesh):
        super().__init__(mesh)
        self.A = self._compute_diffusion_magnitude()
        self.in_vols_pairs = None
        self.Ns = None

    def interpolate(self):
        self._set_internal_vols_pairs()
        self._set_normal_vectors()

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
            ws, vols_idx, neu_ws = self._interpolate_neumann_nodes(
                neumann_nodes)
            vols_idx_flat = np.concatenate(vols_idx)
            ws_flat = np.concatenate(ws)

            ns = np.array([len(vols) for vols in vols_idx])
            nodes_idx = np.repeat(neumann_nodes, ns)

            W_neu = csr_matrix((ws_flat, (nodes_idx, vols_idx_flat)), shape=(
                len(all_nodes), len(self.mesh.volumes)))
        else:
            W_neu = csr_matrix((len(all_nodes), len(self.mesh.volumes)))

        W = W_in + W_neu

        return W

    def _interpolate_internal_nodes(self, in_nodes):
        vols_around_in_nodes = self.mesh.nodes.bridge_adjacencies(
            in_nodes, 1, 3)

        in_faces = self.mesh.faces.internal[:]
        faces_around_in_nodes = self.mesh.nodes.bridge_adjacencies(
            in_nodes, 1, 2)
        in_faces_around_in_nodes = [
            np.intersect1d(f, in_faces, assume_unique=True)
            for f in faces_around_in_nodes]

        ws = []

        for v, KSetv, Sv in zip(
                in_nodes, vols_around_in_nodes, in_faces_around_in_nodes):
            nK = len(KSetv)
            nS = len(Sv)

            xv = self.mesh.nodes.coords[v]

            Ks_Sv = self.mesh.faces.bridge_adjacencies(Sv, 2, 3)

            Mv = np.zeros((nK + 3 * nS, 3 * nK + 1))
            Nv = np.zeros((nK + 3 * nS, nK))

            meq = 0

            for Ki in KSetv:
                xKi = self.mesh.volumes.center[Ki].flatten()

                Mv[meq, (3 * meq):(3 * (meq + 1))] = xKi - xv
                Mv[meq, (3 * nK)] = 1
                Nv[meq, meq] = 1

                meq += 1

            for sj, Kj in zip(Sv, Ks_Sv):
                K1, K2 = Kj[0], Kj[1]
                i_j1 = np.where(KSetv == K1)[0][0]
                i_j2 = np.where(KSetv == K2)[0][0]

                L1, L2 = self.mesh.permeability[Kj].reshape((2, 3, 3))

                eta_j = np.max(self.A[Kj])

                x_sj = self.mesh.faces.center[sj]
                sj_nodes = self.mesh.faces.connectivities[sj]
                sj_nodes_coords = self.mesh.nodes.coords[sj_nodes]
                I, J, K = sj_nodes_coords[0], sj_nodes_coords[1], sj_nodes_coords[2]

                N_sj = np.cross(I - J, I - K)
                n_sj = N_sj / np.linalg.norm(N_sj)

                t_sj1 = xv - x_sj
                t_sj2 = np.cross(n_sj, t_sj1)

                tau_j2 = np.linalg.norm(t_sj2) ** (-eta_j)

                meq += 1
                Mv[meq - 1, (3 * i_j1):(3 * (i_j1 + 1))] = - n_sj @ L1
                Mv[meq - 1, (3 * i_j2):(3 * (i_j2 + 1))] = n_sj @ L2

                meq += 1
                Mv[meq - 1, (3 * i_j1):(3 * (i_j1 + 1))] = - t_sj1
                Mv[meq - 1, (3 * i_j2):(3 * (i_j2 + 1))] = t_sj1

                meq += 1
                Mv[meq - 1, (3 * i_j1):(3 * (i_j1 + 1))] = - tau_j2 * t_sj2
                Mv[meq - 1, (3 * i_j2):(3 * (i_j2 + 1))] = tau_j2 * t_sj2

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
            meq = 0
            nK = len(KSetv)
            nS = len(Sv)
            nb = len(Svb)

            Mb = np.zeros((nK + 3 * nS + nb, 3 * nK + 1))
            Nb = np.zeros((nK + 3 * nS + nb, nK + 1))

            xv = self.mesh.nodes.coords[v]

            Ks_Sv = self.mesh.faces.bridge_adjacencies(Sv, 2, 3)
            Ks_Svb = self.mesh.faces.bridge_adjacencies(Svb, 2, 3)

            if len(Sv) == 1:
                Ks_Sv = Ks_Sv.reshape((1, len(Ks_Sv)))

            for Ki in KSetv:
                xKi = self.mesh.volumes.center[Ki].flatten()

                Mb[meq, (3 * meq):(3 * (meq + 1))] = xKi - xv
                Mb[meq, (3 * nK)] = 1
                Nb[meq, meq] = 1

                meq += 1

            for sj, Kj in zip(Sv, Ks_Sv):
                K1, K2 = Kj[0], Kj[1]
                i_j1 = np.where(KSetv == K1)[0][0]
                i_j2 = np.where(KSetv == K2)[0][0]

                L1, L2 = self.mesh.permeability[Kj].reshape((2, 3, 3))

                eta_j = np.max(self.A[Kj])

                x_sj = self.mesh.faces.center[sj]
                sj_nodes = self.mesh.faces.connectivities[sj]
                sj_nodes_coords = self.mesh.nodes.coords[sj_nodes]
                I, J, K = sj_nodes_coords[0], sj_nodes_coords[1], sj_nodes_coords[2]

                N_sj = np.cross(I - J, I - K)
                n_sj = N_sj / np.linalg.norm(N_sj)

                t_sj1 = xv - x_sj
                t_sj2 = np.cross(n_sj, t_sj1)

                tau_j2 = np.linalg.norm(t_sj2) ** (-eta_j)

                meq += 1
                Mb[meq - 1, (3 * i_j1):(3 * (i_j1 + 1))] = - n_sj @ L1
                Mb[meq - 1, (3 * i_j2):(3 * (i_j2 + 1))] = n_sj @ L2

                meq += 1
                Mb[meq - 1, (3 * i_j1):(3 * (i_j1 + 1))] = - t_sj1
                Mb[meq - 1, (3 * i_j2):(3 * (i_j2 + 1))] = t_sj1

                meq += 1
                Mb[meq - 1, (3 * i_j1):(3 * (i_j1 + 1))] = - tau_j2 * t_sj2
                Mb[meq - 1, (3 * i_j2):(3 * (i_j2 + 1))] = tau_j2 * t_sj2

            for svb, Kb in zip(Svb, Ks_Svb):
                ik = np.where(KSetv == Kb)[0][0]
                L = self.mesh.permeability[Kb].reshape((3, 3))

                gN = self.mesh.neumann[svb][0, 0]

                svb_nodes = self.mesh.faces.connectivities[svb]
                svb_nodes_coords = self.mesh.nodes.coords[svb_nodes]
                I, J, K = svb_nodes_coords[0], svb_nodes_coords[1], svb_nodes_coords[2]

                N_svb = np.cross(I - J, I - K)
                n_svb = N_svb / np.linalg.norm(N_svb)

                meq += 1
                Mb[meq - 1, (3 * ik):(3 * (ik + 1))] = - n_svb @ L
                Nb[meq - 1, nK] = gN

            M = np.linalg.inv(Mb.T @ Mb) @ (Mb.T @ Nb)
            w = M[-1, :]

            wi = w[:-1]
            wc = w[-1]

            ws.append(wi)
            neu_ws.append(wc)

        return ws, vols_around_neu_nodes, neu_ws

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

    def _set_normal_vectors(self):
        """Set the attribute `Ns` which stores the normal vectors 
        to the internal faces.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Retrieve the internal faces.
        internal_faces = self.mesh.faces.internal[:]

        # Retrieve the points that form the components of the normal vectors.
        internal_faces_nodes = self.mesh.faces.bridge_adjacencies(
            internal_faces,
            0, 0)
        I_idx = internal_faces_nodes[:, 0]
        J_idx = internal_faces_nodes[:, 1]
        K_idx = internal_faces_nodes[:, 2]

        I = self.mesh.nodes.coords[I_idx]
        J = self.mesh.nodes.coords[J_idx]
        K = self.mesh.nodes.coords[K_idx]

        n_vols_pairs = len(internal_faces)
        internal_volumes_centers_flat = self.mesh.volumes.center[self.in_vols_pairs.flatten(
        )]
        internal_volumes_centers = internal_volumes_centers_flat.reshape((
            n_vols_pairs,
            2, 3))

        LJ = J - internal_volumes_centers[:, 0]

        # Set the normal vectors.
        Ns = np.cross(I - J, K - J)
        self.Ns_norm = np.linalg.norm(Ns, axis=1)
        self.Ns = Ns / self.Ns_norm[:, np.newaxis]

        # N_sign = np.sign(np.einsum("ij,ij->i", LJ, self.Ns))
        # (self.in_vols_pairs[N_sign < 0, 0],
        #  self.in_vols_pairs[N_sign < 0, 1]) = (self.in_vols_pairs[N_sign < 0, 1],
        #                                        self.in_vols_pairs[N_sign < 0, 0])

    def _compute_diffusion_magnitude(self):
        nvols = len(self.mesh.volumes)
        Ks = self.mesh.permeability[:].reshape((nvols, 3, 3))

        detKs = np.linalg.det(Ks)
        trKs = np.trace(Ks, axis1=1, axis2=2)

        A = (1 - (3 * (detKs ** (1 / 3)) / trKs)) ** 2

        return A
