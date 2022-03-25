import numpy as np
from scipy.sparse import lil_matrix


class MpfadScheme(object):
    def __init__(self, mesh, interpolation):
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

        self._assign_tpfa_terms()

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
        internal_volumes_pairs_flat = self.in_vols_pairs.flatten()

        n_vols_pairs = len(internal_faces)

        internal_faces_centers = self.mesh.faces.center[internal_faces]
        internal_volumes_centers_flat = self.mesh.volumes.center[internal_volumes_pairs_flat]
        internal_volumes_centers = internal_volumes_centers_flat.reshape((
            n_vols_pairs,
            2, 3))

        self.h_L = np.linalg.norm(
            internal_volumes_centers[:, 0, :] - internal_faces_centers, axis=1)
        self.h_R = np.linalg.norm(
            internal_volumes_centers[:, 1, :] - internal_faces_centers, axis=1)

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
        Is_idx = self.mesh.faces.connectivities[internal_faces][:, 0]
        Js_idx = self.mesh.faces.connectivities[internal_faces][:, 1]

        Is = self.mesh.nodes.coords[Is_idx]
        Js = self.mesh.nodes.coords[Js_idx]

        internal_faces_centers = self.mesh.faces.center[internal_faces]

        # Set the normal vectors.
        self.Ns = np.cross(internal_faces_centers - Is,
                           internal_faces_centers - Js)
        self.Ns_norm = np.linalg.norm(self.Ns, axis=1)

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

        V = np.hstack((self.Ns, tau_ij)).reshape((len(self.Ns) * 2, 3))

        K_all = self.mesh.permeability[internal_volumes_pairs_flat].reshape(
            (n_vols_pairs * 2, 3, 3))

        Kt_ij_partial = np.einsum("ij,ikj->ik", V, K_all)
        Kt_ij_flat = np.einsum("ij,ij->i", Kt_ij_partial,
                               V) / (np.linalg.norm(V, axis=1) ** 2)
        Kt_ij_all = Kt_ij_flat.reshape((n_vols_pairs, 2))

        Kt_ij_L = Kt_ij_all[:, 0]
        Kt_ij_R = Kt_ij_all[:, 1]

        return Kt_ij_L, Kt_ij_R

    def _assign_tpfa_terms(self):
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
        self.A[self.in_vols_pairs[:, 0],
               self.in_vols_pairs[:, 1]] = -faces_trans[:]
        self.A[self.in_vols_pairs[:, 1],
               self.in_vols_pairs[:, 0]] = -faces_trans[:]

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
        in_vols_pairs_flat = self.in_vols_pairs.flatten(order="F")

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
