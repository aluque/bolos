""" Routines to handle different kinds of grids (linear, quadratic, logarithmic)
"""
import numpy as np
from scipy.interpolate import interp1d

class Grid(object):
    """ Class to define energy grids.

    This class encapsulates the information about an energy grid.

    Parameters
    ----------
    x0 : float
       Lowest boundary energy.
    x1 : float
       Highest energy boundary.
    n : float
       Number of cells

    Notes
    -----
    This is a base class and you usually do not want to instantiate it
    directly.  You can define new grid classes by subclassing this class and
    then defining an `f` method that maps energy to a new variable `y`
    that is divided uniformly.

    See Also
    --------
    LinearGrid : A grid with linear spacings (constant cell length).
    QuadraticGrid : A grid with quadratic spacings (linearly increasing 
       cell length).
    GeometricGrid : A grid with geometrically increasing cell lengths.
    LogGrid : A logarithmic grid.

    """
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
        
        # This is useful in some routines that integrate eps**1/2 * f
        self.d32 = self.b[1:]**1.5 - self.b[:-1]**1.5

        self._interp = None


    def interpolate(self, f, other):
        """ Interpolates into this grid an eedf defined in another grid. 

        Parameters
        ----------
        f : array or array-like
           The original EEDF
        other : :class:`Grid`
           The old grid, where `f` is defined.

        Returns
        -------
        fnew : array or array-like
           An EEDF defined in our grid.


        """
        if self._interp is None:
            # Sould we extrapolate linearly instead od by closest value?
            self._interp = interp1d(np.r_[other.x0, other.c, other.x1],
                                    np.r_[f[0], f, f[-1]],
                                    bounds_error=False, fill_value=0)
        
        return self._interp(self.c)



    def cell(self, x):
        """ Returns the cell index containing the value x. 

        Parameters
        ----------
        x : float
           The value x which you want to localize.

        Returns
        -------
        index : int
           The index to the cell containing x

        """
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
        return np.sqrt(x - self.x0)

    def finv(self, w):
        return w**2 + self.x0


class GeometricGrid(Grid):
    """ A grid with geometrically progressing spacing. To be more precise, 
    here the length
    of cell i+1 is r times the length of cell i.
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
    def __init__(self, x0, x1, n, s=10.):
        self.s = s
        super(LogGrid, self).__init__(x0, x1, n)


    def f(self, x):
        return np.log(self.s + (x - self.x0))


    def finv(self, w):
        return np.exp(w) - self.s + self.x0


class AutomaticGrid(Grid):
    """ A grid set automatically using a previous estimation of the EEDF
    to fix a peak energy.  """
    def __init__(self, grid, f0, delta=1e-4):
        # We will create a new grid where the number of particles is roughly
        # the same inside each cell and the number of cells is the same as in
        # grid.
        
        # TODO: This is also calculated in the solver.BoltzmannSolver class
        cum = np.r_[0.0, np.cumsum(grid.d32 * f0)]


        # If we had integrated f0 exactly, cum[-1] would be 1.  However it may
        # be slightly differentso we will renormalize here to prevent an error
        # when we interpolate for a number very close to 1.0.
        cum[:] = cum / cum[-1]
        
        interp = interp1d(cum, grid.b)
        nnew = np.linspace(0.0, 1.0, grid.n + 1)

        self.n, self.x0, self.x1 = grid.n, grid.x0, grid.x1

        self.b = interp(nnew)

        self.c = 0.5 * (self.b[1:] + self.b[:-1])
        self.d = np.diff(self.b)

        self._interp = None



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


