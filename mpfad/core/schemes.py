import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from ..interpolations.base import BaseInterpolation


class MpfadScheme(object):
    def __init__(self, mesh, interpolation: BaseInterpolation):
        self.mesh = mesh
        self.interpolation = interpolation

        n = len(self.mesh.volumes)
        self.A = lil_matrix((n, n))
        self.q = np.zeros(n)

        # Pairs of volumes sharing an internal face.
        self.in_vols_pairs = None

        # Distances from the internal faces to the volumes who share it.
        self.h_L = None
        self.h_R = None

        # Normal vectors to internal faces.
        self.Ns = None
        self.Ns_norm = None

        # Normal projections of the permeability tensor.
        self.Kn_L = None
        self.Kn_R = None

    def assemble(self):
        """Assemble the transmissibility of the MPFA-D scheme. After the call, the
        attributes `A` and `q` are set and can be used to solve the problem.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._set_internal_vols_pairs()
        self._set_normal_vectors()
        self._set_normal_distances()
        self._set_normal_permeabilities()

        A_tpfa = self._assemble_tpfa_matrix()
        A_cdt = self._assemble_cdt_matrix()

        A_D, q_D = self._handle_dirichlet_bc()
        q_N = self._handle_neumann_bc()

        A = A_tpfa + A_cdt + A_D
        q = q_D + q_N

        return A, q

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

    def _set_normal_distances(self):
        """Compute the distances from the center of the internal faces to their
        adjacent volumes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        internal_faces = self.mesh.faces.internal[:]

        L = self.mesh.volumes.center[self.in_vols_pairs[:, 0]]
        R = self.mesh.volumes.center[self.in_vols_pairs[:, 1]]

        J_idx = self.mesh.faces.connectivities[internal_faces][:, 1]
        J = self.mesh.nodes.coords[J_idx]

        LJ = J - L
        LR = J - R

        self.h_L = np.abs(np.einsum("ij,ij->i", self.Ns, LJ) / self.Ns_norm)
        self.h_R = np.abs(np.einsum("ij,ij->i", self.Ns, LR) / self.Ns_norm)

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
        I_idx = self.mesh.faces.connectivities[internal_faces][:, 0]
        J_idx = self.mesh.faces.connectivities[internal_faces][:, 1]
        K_idx = self.mesh.faces.connectivities[internal_faces][:, 2]

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
        self.Ns = np.cross(I - J, K - J)
        self.Ns_norm = np.linalg.norm(self.Ns, axis=1)

        N_sign = np.sign(np.einsum("ij,ij->i", LJ, self.Ns))
        (self.in_vols_pairs[N_sign < 0, 0],
         self.in_vols_pairs[N_sign < 0, 1]) = (self.in_vols_pairs[N_sign < 0, 1],
                                               self.in_vols_pairs[N_sign < 0, 0])

    def _set_normal_permeabilities(self):
        """Compute the normal projections of the permeability tensors.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        n_vols_pairs = len(self.mesh.faces.internal)

        internal_volumes_pairs_flat = self.in_vols_pairs.flatten(order="F")

        K_all = self.mesh.permeability[internal_volumes_pairs_flat].reshape(
            (n_vols_pairs * 2, 3, 3))
        N_dup = np.hstack((self.Ns, self.Ns)).reshape((len(self.Ns) * 2, 3))
        Kn_partial = np.einsum("ij,ikj->ik", N_dup, K_all)
        Kn_all_part = np.einsum("ij,ij->i", Kn_partial,
                                N_dup) / (np.linalg.norm(N_dup, axis=1) ** 2)
        Kn_all = Kn_all_part.reshape((n_vols_pairs, 2))

        self.Kn_L = Kn_all[:, 0]
        self.Kn_R = Kn_all[:, 1]

    def _compute_tangent_permeabilities(self, tau_ij):
        """Computes the tangent projection of the permeability tensors
        given vectors `tau_ij`.

        Parameters
        ----------
        tau_ij: A N x 3 numpy array representing stacked vectors.

        Returns
        -------
        A tuple of arrays containing the projections to the left and
        to the right of the internal faces.
        """
        n_vols_pairs = len(self.mesh.faces.internal)
        internal_volumes_pairs_flat = self.in_vols_pairs.flatten(order="F")

        tau_ij_dup = np.hstack((tau_ij, tau_ij)).reshape((len(self.Ns) * 2, 3))

        Ns_dup = np.hstack((self.Ns, self.Ns)).reshape((len(self.Ns) * 2, 3))
        Ns_norm_dup = np.linalg.norm(Ns_dup, axis=1)

        K_all = self.mesh.permeability[internal_volumes_pairs_flat].reshape(
            (n_vols_pairs * 2, 3, 3))

        Kt_ij_partial = np.einsum("ij,ikj->ik", Ns_dup, K_all)
        Kt_ij_flat = np.einsum("ij,ij->i", Kt_ij_partial,
                               tau_ij_dup) / (Ns_norm_dup ** 2)
        Kt_ij_all = Kt_ij_flat.reshape((n_vols_pairs, 2))

        Kt_ij_L = Kt_ij_all[:, 0]
        Kt_ij_R = Kt_ij_all[:, 1]

        return Kt_ij_L, Kt_ij_R

    def _assemble_tpfa_matrix(self):
        """Set the TPFA terms of the transsmissibility matrix `A`.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Compute the face transmissibilities.
        Kn_prod = self.Kn_L * self.Kn_R
        Keq = Kn_prod / ((self.Kn_L * self.h_R) +
                         (self.Kn_R * self.h_L))
        faces_trans = Keq * self.Ns_norm

        # Set transmissibilities in matrix.
        n_vols = len(self.mesh.volumes)

        data = np.hstack((-faces_trans, -faces_trans))
        row_idx = np.hstack((self.in_vols_pairs[:, 0], self.in_vols_pairs[:, 1]))
        col_idx = np.hstack((self.in_vols_pairs[:, 1], self.in_vols_pairs[:, 0]))

        A_tpfa = csr_matrix((data, (row_idx, col_idx)), shape=(n_vols, n_vols))

        A_tpfa.setdiag(-A_tpfa.sum(axis=1))

        return A_tpfa

    def _compute_cdt_terms(self):
        """Compute the cross diffusion terms of the MPFA-D scheme.

        Parameters
        ----------
        None

        Returns
        -------
        A tuple of numpy arrays containing the terms D_JK and D_JI.
        """
        n_vols_pairs = len(self.mesh.faces.internal)

        in_vols_pairs_flat = self.in_vols_pairs.flatten()
        in_vols_centers_flat = self.mesh.volumes.center[in_vols_pairs_flat]
        in_vols_centers = in_vols_centers_flat.reshape((n_vols_pairs, 2, 3))

        LR = in_vols_centers[:, 1, :] - in_vols_centers[:, 0, :]

        internal_faces = self.mesh.faces.internal[:]

        I_idx = self.mesh.faces.connectivities[internal_faces][:, 0]
        J_idx = self.mesh.faces.connectivities[internal_faces][:, 1]
        K_idx = self.mesh.faces.connectivities[internal_faces][:, 2]

        I = self.mesh.nodes.coords[I_idx]
        J = self.mesh.nodes.coords[J_idx]
        K = self.mesh.nodes.coords[K_idx]

        tau_JK = np.cross(self.Ns, J - K)
        tau_JI = np.cross(self.Ns, J - I)

        Kt_JK_L, Kt_JK_R = self._compute_tangent_permeabilities(tau_JK)
        Kt_JI_L, Kt_JI_R = self._compute_tangent_permeabilities(tau_JI)

        A1_JK = np.einsum("ij,ij->i", tau_JK, LR) / (self.Ns_norm ** 2)
        A2_JK = (self.h_L * (Kt_JK_L / self.Kn_L) + self.h_R *
                 (Kt_JK_R / self.Kn_R)) / self.Ns_norm
        D_JK = A1_JK - A2_JK

        A1_JI = np.einsum("ij,ij->i", tau_JI, LR) / (self.Ns_norm ** 2)
        A2_JI = (self.h_L * (Kt_JI_L / self.Kn_L) + self.h_R *
                 (Kt_JI_R / self.Kn_R)) / self.Ns_norm
        D_JI = A1_JI - A2_JI

        return D_JK, D_JI

    def _assemble_cdt_matrix(self):
        """Assign the cross diffusion terms to the transmissibility matrix.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Compute the cross diffusion coefficients.
        D_JK, D_JI = self._compute_cdt_terms()

        # Compute the weights for the interpolation of the pressure in a node.
        W = self.interpolation.interpolate()

        # Find the connectivities of the internal faces.
        in_faces = self.mesh.faces.internal
        in_faces_nodes = self.mesh.faces.connectivities[in_faces]
        I, J, K = in_faces_nodes[:,
                                 0], in_faces_nodes[:, 1], in_faces_nodes[:, 2]

        # Compute the face transmissibilities.
        Kn_prod = self.Kn_L * self.Kn_R
        Keq = Kn_prod / ((self.Kn_L * self.h_R) +
                         (self.Kn_R * self.h_L))
        Keq_N = self.Ns_norm * Keq

        # Assemble the CDT matrix.
        A_cdt = csr_matrix((len(self.mesh.volumes), len(self.mesh.volumes)))

        cdt_partial = 0.5 * ((W[J, :] - W[I, :]).multiply(D_JK[:, np.newaxis]) - (
            W[J, :] - W[K, :]).multiply(D_JI[:, np.newaxis])).multiply(Keq_N[:, np.newaxis])
        cdt = cdt_partial.tocsr()

        A_cdt[self.in_vols_pairs[:, 0], :] += cdt
        A_cdt[self.in_vols_pairs[:, 1], :] += cdt

        return A_cdt

    def _handle_dirichlet_bc(self):
        bfaces = self.mesh.faces.boundary[:]
        bfaces_dirichlet_values = self.mesh.dirichlet_faces[bfaces].flatten()
        dirichlet_faces = bfaces[bfaces_dirichlet_values == 1]

        dirichlet_nodes = self.mesh.faces.connectivities[dirichlet_faces]
        dirichlet_volumes = self.mesh.faces.bridge_adjacencies(
            dirichlet_faces, 2, 3).flatten()

        R = self.mesh.volumes.center[dirichlet_volumes]
        dirichlet_faces_centers = self.mesh.faces.center[dirichlet_faces]
        I_idx, J_idx, K_idx = (
            dirichlet_nodes[:, 0],
            dirichlet_nodes[:, 1],
            dirichlet_nodes[:, 2])
        I, J, K = (
            self.mesh.nodes.coords[I_idx],
            self.mesh.nodes.coords[J_idx],
            self.mesh.nodes.coords[K_idx])

        N = np.cross(dirichlet_faces_centers - I,
                     dirichlet_faces_centers - J)
        N_norm = np.linalg.norm(N, axis=1)

        tau_JK = np.cross(N, J - K)
        tau_JI = np.cross(N, J - I)

        JR = J - R
        h_R = np.linalg.norm(R - dirichlet_faces_centers, axis=1)

        K_all = self.mesh.permeability[dirichlet_volumes].reshape(
            (len(dirichlet_volumes), 3, 3))

        Kn_R_partial = np.einsum("ij,ikj->ik", N, K_all)
        Kn_R = np.einsum("ij,ij->i", Kn_R_partial, N) / (N_norm ** 2)

        Kt_JK = np.einsum("ij,ij->i", Kn_R_partial, tau_JK) / (N_norm ** 2)

        Kt_JI = np.einsum("ij,ij->i", Kn_R_partial, tau_JI) / (N_norm ** 2)

        D_JI = -(np.einsum("ij,ij->i", tau_JK, JR)
                 * Kn_R) / (N_norm * h_R) + Kt_JK
        D_JK = -(np.einsum("ij,ij->i", tau_JI, JR)
                 * Kn_R) / (N_norm * h_R) + Kt_JI

        gD = self.mesh.dirichlet_nodes[dirichlet_nodes.flatten()].reshape(
            dirichlet_nodes.shape[0], 3)
        gD_I, gD_J, gD_K = gD[:, 0], gD[:, 1], gD[:, 2]

        diag_A_D = np.zeros(len(self.mesh.volumes))
        np.add.at(diag_A_D, dirichlet_volumes, -2 * (Kn_R / h_R))

        A_D = csr_matrix((len(self.mesh.volumes), len(self.mesh.volumes)))
        A_D.setdiag(diag_A_D)

        q_D = np.zeros(len(self.mesh.volumes))
        np.add.at(q_D, dirichlet_volumes, -2 * (Kn_R * gD_J / h_R) + D_JI * (
            gD_J - gD_I) + D_JK * (gD_J - gD_K))

        return A_D, q_D

    def _handle_neumann_bc(self):
        bfaces = self.mesh.faces.boundary[:]
        bfaces_neumann_values = self.mesh.neumann[bfaces].flatten()
        neumann_faces = bfaces[bfaces_neumann_values != 0]
        neumann_values = bfaces_neumann_values[bfaces_neumann_values != 0]

        q_N = np.zeros(len(self.mesh.volumes))

        if len(neumann_faces) > 0:
            neumann_volumes = self.mesh.faces.bridge_adjacencies(
                neumann_faces, 2, 3).flatten()
            np.add.at(q_N, neumann_volumes, neumann_values)

        return q_N
