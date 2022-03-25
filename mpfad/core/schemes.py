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

    def set_cdt_terms(self):
        pass
