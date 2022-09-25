from mpfad.core.schemes import MpfadScheme
from scipy.sparse.linalg import splu, spsolve
from scipy.sparse import diags, csc_matrix
import numpy as np


class BaseNonLinearCorrection(object):
    def __init__(self, mesh, mpfad_scheme: MpfadScheme):
        self.mesh = mesh
        self.mpfad = mpfad_scheme

    def run(self, assemble_mpfad_matrix=True):
        if assemble_mpfad_matrix:
            self.mpfad.assemble()


class MpfadNonLinearDefectionCorrection(BaseNonLinearCorrection):
    def __init__(self, mesh, mpfad_scheme: MpfadScheme, dmp_tol=1e-6):
        super().__init__(mesh, mpfad_scheme)
        self.dmp_tol = dmp_tol
        all_vols = self.mesh.volumes.all[:]
        self.vols_neighbors_by_node = self.mesh.volumes.bridge_adjacencies(
            all_vols,
            0, 3)
        self.L_tpfa = None
        self.D_tpfa = None
        self.U_tpfa = None

    def run(self, assemble_mpfad_matrix=True):
        super().run(assemble_mpfad_matrix)

    def _compute_max_min_solutions_in_stencil(self, ut):
        all_vols = self.mesh.volumes.all[:]
        vols_neighbors_flat = np.concatenate(self.vols_neighbors_by_node)

        ns = np.array([neigh.shape[0] for neigh in self.vols_neighbors_by_node])
        vols_neighbors_idx = np.repeat(all_vols, ns)

        Ut_mat = csc_matrix(
            (ut[vols_neighbors_flat],
             (vols_neighbors_idx, vols_neighbors_flat)))

        ut_max = Ut_mat.max(axis=1).toarray().flatten()
        ut_min = Ut_mat.min(axis=1).toarray().flatten()

        return ut_max, ut_min

    def _find_correction_params_intervals(
            self, ut, ut_max, ut_min, q_tpfa, q_cdt, L_cdt, D_cdt, U_cdt):
        x_max = (U_cdt @ ut) + (D_cdt @ ut_max) + (L_cdt @ ut_max) - q_cdt
        y_max = (self.U_tpfa @ ut) + (self.D_tpfa @
                                      ut_max) + (self.L_tpfa @ ut_max) - q_tpfa
        x_min = (U_cdt @ ut) + (D_cdt @ ut_min) + (L_cdt @ ut_min) - q_cdt
        y_min = (self.U_tpfa @ ut) + (self.D_tpfa @
                                      ut_min) + (self.L_tpfa @ ut_min) - q_tpfa

        Y_max = np.zeros((len(self.mesh.volumes), 2))
        Y_min = np.zeros((len(self.mesh.volumes), 2))

        Y_max[x_max > 0, 0] = -y_max[x_max > 0] / x_max[x_max > 0]
        Y_max[x_max > 0, 1] = np.inf

        Y_max[x_max < 0, 0] = -np.inf
        Y_max[x_max < 0, 1] = -y_max[x_max < 0] / x_max[x_max < 0]

        Y_max[x_max == 0, 0] = 0
        Y_max[x_max == 0, 1] = 1

        Y_min[x_min > 0, 0] = -np.inf
        Y_min[x_min > 0, 1] = y_min[x_min > 0] / x_min[x_min > 0]

        Y_min[x_min < 0, 0] = -y_min[x_min < 0] / x_min[x_min < 0]
        Y_min[x_min < 0, 1] = np.inf

        Y_min[x_min == 0, 0] = 0
        Y_min[x_min == 0, 1] = 1

        Y = np.zeros((len(self.mesh.volumes), 2))

        ut_min_mask = ut <= ut_min
        Y[ut_min_mask, 0] = np.maximum(Y_min[ut_min_mask, 0], 0)
        Y[ut_min_mask, 1] = np.minimum(Y_min[ut_min_mask, 1], 1)

        ut_max_mask = (ut > ut_min) & (ut >= ut_max)
        Y[ut_max_mask, 0] = np.maximum(Y_max[ut_max_mask, 0], 0)
        Y[ut_max_mask, 1] = np.minimum(Y_max[ut_max_mask, 1], 1)

        else_mask = ~ut_min_mask & ~ut_max_mask
        Y[else_mask, 0] = np.maximum(np.maximum(
            Y_min[else_mask, 0], Y_max[else_mask, 0]), 0)
        Y[else_mask, 1] = np.minimum(np.minimum(
            Y_min[else_mask, 1], Y_max[else_mask, 1]), 1)

        empty_Yi_mask = Y[:, 0] >= Y[:, 1]
        Y[empty_Yi_mask, :] = [0, 1]

        return Y

    def _compute_correction_params(self, Y, ut, ut_max, ut_min):
        # Correction coef. intervals for each face.
        F = np.zeros((self.mpfad.in_vols_pairs.shape[0], 2))

        L_idx = self.mpfad.in_vols_pairs[:, 0]
        R_idx = self.mpfad.in_vols_pairs[:, 1]

        Y_L = Y[L_idx]
        Y_R = Y[R_idx]

        F[:, 0] = np.maximum(Y_R[:, 0], Y_L[:, 0])
        F[:, 1] = np.minimum(Y_R[:, 1], Y_L[:, 1])

        # Check if the correction interval for a face is empty.
        empty_F_mask = F[:, 0] >= F[:, 1]

        # Check which volume sharing the face violates the DMP at the
        # current iteration.
        R_dmp_viol_mask = (ut[R_idx] < (
            ut_min[R_idx] - self.dmp_tol)) | (ut[R_idx] > (ut_max[R_idx] + self.dmp_tol))
        L_dmp_viol_mask = (ut[L_idx] < (
            ut_min[L_idx] - self.dmp_tol)) | (ut[L_idx] > (ut_max[L_idx] + self.dmp_tol))

        F[empty_F_mask & L_dmp_viol_mask] = Y_L[empty_F_mask & L_dmp_viol_mask]
        F[empty_F_mask & R_dmp_viol_mask] = Y_R[empty_F_mask & R_dmp_viol_mask]

        else_mask = empty_F_mask & R_dmp_viol_mask & L_dmp_viol_mask
        F[else_mask, 0] = 0.5 * (Y_L[else_mask, 0] + Y_R[else_mask, 0])
        F[else_mask, 1] = 0.5 * (Y_L[else_mask, 1] + Y_R[else_mask, 1])

        # Find the correction parameters.
        alpha = np.zeros(len(self.mesh.faces))

        max_F_mask = F[:, 1] < 1
        alpha[max_F_mask] = F[max_F_mask, 1]
        alpha[~max_F_mask] = 0.5 * np.sum(F[~max_F_mask], axis=1)

        return alpha

    def _update_cdt(self, alpha, At_cdt, bt_cdt, Tt_cdt, Ft_cdt):
        diag_alpha = diags(alpha - 1)
        At_next_cdt = At_cdt + diag_alpha @ self.mpfad.D @ Tt_cdt
        bt_next_cdt = bt_cdt + diag_alpha @ self.mpfad.D @ Ft_cdt
        return At_next_cdt, bt_next_cdt

    def _compute_ldu_decomposition(self, M):
        lu_decomp = splu(M)
        diag_U = lu_decomp.U.diagonal()
        L = lu_decomp.L
        D = diags(diag_U)
        U = lu_decomp.U / diag_U[:, None]

        return L, D, U

    def _compute_l2_error(self, u1, u2):
        V = self.mesh.volumes.volume[:]
        l2_err = (np.sum(((u1 - u2) ** 2) * V) / np.sum((u1 ** 2) * V)) ** 0.5
        return l2_err
