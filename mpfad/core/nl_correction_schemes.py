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
