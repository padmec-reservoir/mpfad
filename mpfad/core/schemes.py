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
        pass

    def set_tpfa_terms(self):
        pass

    def set_cdt_terms(self):
        pass
