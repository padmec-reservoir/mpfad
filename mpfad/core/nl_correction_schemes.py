from mpfad.core.schemes import MpfadScheme


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
