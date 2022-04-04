class BaseInterpolation(object):
    def __init__(self, mesh):
        self.mesh = mesh

    def interpolate(self):
        raise NotImplementedError()
