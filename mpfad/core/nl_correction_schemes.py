from mpfad.core.schemes import MpfadScheme


class BaseNonLinearCorrection(object):
    def __init__(self, mesh, mpfad_scheme: MpfadScheme):
        self.mesh = mesh
        self.mpfad = mpfad_scheme

    def run(self, assemble_mpfad_matrix=True):
        if assemble_mpfad_matrix:
            self.mpfad.assemble()
