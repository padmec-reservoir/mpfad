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

    def run(self, assemble_mpfad_matrix=True):
        super().run(assemble_mpfad_matrix)

    def _find_correction_params_intervals(
            self, ut, ut_max, ut_min, q_tpfa, q_cdt, L_tpfa, D_tpfa, U_tpfa,
            L_cdt, D_cdt, U_cdt):
        x_max = (U_cdt @ ut) + (D_cdt @ ut_max) + (L_cdt @ ut_max) - q_cdt
        y_max = (U_tpfa @ ut) + (D_tpfa @ ut_max) + (L_tpfa @ ut_max) - q_tpfa
        x_min = (U_cdt @ ut) + (D_cdt @ ut_min) + (L_cdt @ ut_min) - q_cdt
        y_min = (U_tpfa @ ut) + (D_tpfa @ ut_min) + (L_tpfa @ ut_min) - q_tpfa

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
