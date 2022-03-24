import numpy as np
from scipy.sparse import lil_matrix


class MpfadScheme(object):
    def __init__(self, mesh, interpolation):
        self.mesh = mesh
        self.interpolation = interpolation

        n = len(self.mesh.volumes)
        self.A = lil_matrix((n, n))
        self.q = np.zeros(n)

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
        self.set_tpfa_terms()

    def set_tpfa_terms(self):
        """Set the TPFA terms of the transsmissibility matrix `A`.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Retrieve the internal faces to compute all pairs of volumes.
        internal_faces = self.mesh.faces.internal[:]
        n_vols_pairs = len(internal_faces)

        # Compute the internal faces' normal vectors.
        internal_faces_nodes_1 = self.mesh.faces.connectivities[internal_faces][:, 0]
        internal_faces_nodes_2 = self.mesh.faces.connectivities[internal_faces][:, 1]
        V1 = self.mesh.nodes.coords[internal_faces_nodes_1]
        V2 = self.mesh.nodes.coords[internal_faces_nodes_2]
        internal_faces_centers = self.mesh.faces.center[internal_faces]

        Ns = np.cross(internal_faces_centers - V1, internal_faces_centers - V2)

        # Find the internal volumes, i.e., the volumes containing an internal face.
        internal_volumes_pairs = self.mesh.faces.bridge_adjacencies(
            internal_faces,
            2, 3)
        internal_volumes_pairs_flat = internal_volumes_pairs.flatten()
        internal_volumes_pairs_flat_f = internal_volumes_pairs.flatten(
            order="F")

        # Compute the distance (centroid-centroid distance) between
        # the internal face and the volumes who share it.
        internal_volumes_centers_flat = self.mesh.volumes.center[internal_volumes_pairs_flat]
        internal_volumes_centers = internal_volumes_centers_flat.reshape((
            n_vols_pairs,
            2, 3))

        h_L = np.linalg.norm(
            internal_volumes_centers[:, 0, :] - internal_faces_centers, axis=1)
        h_R = np.linalg.norm(
            internal_volumes_centers[:, 1, :] - internal_faces_centers, axis=1)

        # Compute the orthogonal projections of the permeability tensors.
        K_all = self.mesh.permeability[internal_volumes_pairs_flat_f].reshape(
            (n_vols_pairs * 2, 3, 3))
        N_dup = np.hstack((Ns, Ns)).reshape((len(Ns) * 2, 3))
        K_eq_partial = np.einsum("ij,ikj->ik", N_dup, K_all)
        K_eq_all_part = np.einsum(
            "ij,ij->i", K_eq_partial, N_dup) / (np.linalg.norm(N_dup, axis=1) ** 2)
        K_eq_all = K_eq_all_part.reshape((2, n_vols_pairs))

        # Compute the face transmissibilities.
        K_eq_prod = K_eq_all[0, :] * K_eq_all[1, :]
        faces_trans = K_eq_prod / ((K_eq_all[0, :] * h_R) +
                                   (K_eq_all[1, :] * h_L))

        # Set transmissibilities in matrix.
        self.A[internal_volumes_pairs[:, 0],
               internal_volumes_pairs[:, 1]] = -faces_trans[:]
        self.A[internal_volumes_pairs[:, 1],
               internal_volumes_pairs[:, 0]] = -faces_trans[:]

    def set_cdt_terms(self):
        pass
