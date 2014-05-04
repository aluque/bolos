""" Routines to handle different kinds of grids (linear, quadratic, logarithmic)
"""
import numpy as np

class Grid(object):
    """ Grid class.  Different kinds of grid are defined by a function
    f and its inverse; these define the spacing within the grid.  """
    def __init__(self, x0, x1, n):
        self.x0 = x0
        self.x1 = x1
        self.delta = x1 - x0

        self.fx0 = self.f(x0)
        self.fx1 = self.f(x1)

        self.n = n

        # Boundaries at i - 1/2
        fx = np.linspace(self.fx0, self.fx1, self.n + 1)
        self.b = self.finv(fx)
        
        # centers
        self.c = 0.5 * (self.b[1:] + self.b[:-1])

        # And these are the deltas
        self.d = np.diff(self.b)

        # This is the spacing of the mapped x
        self.df = fx[1] - fx[0]
        
    def cell(self, x):
        """ Returns the cell index containing the value x. """
        return int((self.f(x) - self.fx0) / self.df)
        

class LinearGrid(Grid):
    """ A grid with linear spacing. """
    def f(self, x):
        return x

    def finv(self, w):
        return w


class QuadraticGrid(Grid):
    """ A grid with quadratic spacing. """
    def f(self, x):
        return sqrt(x - self.x0)

    def finv(self, w):
        return w**2 + self.x0


class GeometricGrid(Grid):
    """ A grid with geometrically progressing spacing. To be more precise, 
    here the length
    of cell i+1 is r times the length of cell i.  Perhaps calling this a
    'geometric' grid would be more appropriate.
    """
    def __init__(self, x0, x1, n, r=1.1):
        self.r = r
        self.logr = np.log(r)
        self.rn_minus_1 = np.exp(n * self.logr) - 1

        super(GeometricGrid, self).__init__(x0, x1, n)


    def f(self, x):
        return (np.log(1 + (x - self.x0) * self.rn_minus_1 / self.delta)
                / self.logr)

    def finv(self, w):
        return (self.x0 + self.delta * (np.exp(w * self.logr) - 1) 
                / self.rn_minus_1)


class LogGrid(Grid):
    """ A pseudo-logarithmic grid.  We add a certain s to the variable
    to avoid log(0) = -inf. The grid is actually logarithmic only for
    x >> s.
    """
    def __init__(self, x0, x1, n, s=10):
        self.s = s
        super(LogGrid, self).__init__(x0, x1, n)


    def f(self, x):
        return np.log(self.s + (x - self.x0))


    def finv(self, w):
        return np.exp(w) - self.s + self.x0



def mkgrid(kind, *args, **kwargs):
    """ Builds and returns a grid of class kind.  Possible values are
    'linear', 'lin', 'quadratic', 'quad', 'logarithmic', 'log'.
    """
    GRID_CLASSES = {'linear': LinearGrid,
                    'lin': LinearGrid,
                    'quadratic': QuadraticGrid,
                    'quad': QuadraticGrid,
                    'geometric': GeometricGrid,
                    'geo': GeometricGrid,
                    'logarithmic': LogGrid,
                    'log': LogGrid}

    klass = GRID_CLASSES[kind]

    return klass(*args, **kwargs)


