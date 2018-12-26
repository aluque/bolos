from builtins import str
from builtins import object
import logging

import numpy as np
from scipy.interpolate import interp1d


class Process(object):
    # The factor of in-scatering.  
    IN_FACTOR = {'EXCITATION': 1,
                 'IONIZATION': 2,
                 'ATTACHMENT': 0,
                 'ELASTIC': 1,
                 'MOMENTUM': 1,
                 'EFFECTIVE': 1}

    # The shift factor for inelastic collisions. 
    SHIFT_FACTOR = {'EXCITATION': 1,
                    'IONIZATION': 2,
                    'ATTACHMENT': 1,
                    'ELASTIC': 1,
                    'MOMENTUM': 1,
                    'EFFECTIVE': 1}

                 
    def __init__(self, target=None, kind=None, data=None,
                 comment='', mass_ratio=None,
                 product=None, threshold=0, weight_ratio=None):
        self.target_name = target

        # We will link this later
        self.target = None

        self.kind = kind
        self.data = np.array(data)

        self.x = self.data[:, 0]
        self.y = self.data[:, 1]

        self.comment = comment
        self.mass_ratio = mass_ratio
        self.product = product
        self.threshold = threshold
        self.weight_ratio = weight_ratio
        self.interp = padinterp(self.data)
        self.isnull = False

        
        self.in_factor = self.IN_FACTOR.get(self.kind, None)
        self.shift_factor = self.SHIFT_FACTOR.get(self.kind, None)
        
        if np.amin(self.data[:, 0]) < 0:
            raise ValueError("Negative energy in the cross section %s"
                             % str(self))
 
        if np.amin(self.data[:, 1]) < 0:
            raise ValueError("Negative cross section for %s"
                             % str(self))
       
        self.cached_grid = None


    def scatterings(self, g, eps):
        if len(self.j) == 0:
            # When we do not have inelastic collisions or when the grid is
            # smaller than the thresholds, we still return an empty array
            # and thus avoid exceptions in g[self.j]
            return np.array([], dtype='f')

        gj = g[self.j]
        epsj = eps[self.j]
        r = int_linexp0(self.eps[:, 0], self.eps[:, 1], 
                        self.sigma[:, 0], self.sigma[:, 1],
                        gj, epsj)
        return r
        
        
    def set_grid_cache(self, grid):
        """ Sets a grid cache of the intersections between grid cell j and grid
        cell i shifted. 
        """

        # We will create an arras with matching 
        # rows ([i], [j], [eps1, eps2], [sigma1, sigma2])
        # that contain the overlap between the shifted cell i and cell j.
        # However we may have more than one row for a given i, j if 
        # an interpolation point for sigma falls inside the interval.

        if self.cached_grid is grid:
            # We only have to redo all these computations when the grid changes
            # so we store the grid for which this has been already calculated.
            return

        self.cached_grid = grid

        eps1 = self.shift_factor * grid.b + self.threshold
        eps1[:] = np.maximum(eps1, grid.b[0] + 1e-9)
        eps1[:] = np.minimum(eps1, grid.b[-1] - 1e-9)

        fltb = np.logical_and(grid.b >= eps1[0], grid.b <= eps1[-1])
        fltx = np.logical_and(self.x >= eps1[0], self.x <= eps1[-1])
        nodes = np.unique(np.r_[eps1, grid.b[fltb], self.x[fltx]])


        sigma0 = self.interp(nodes)
        
        self.j = np.searchsorted(grid.b, nodes[1:]) - 1
        self.i = np.searchsorted(eps1, nodes[1:]) - 1
        self.sigma = np.c_[sigma0[:-1], sigma0[1:]]
        self.eps   = np.c_[nodes[:-1], nodes[1:]]

    def __str__(self):
        return "{%s: %s %s}" % (self.kind, self.target_name, 
                                "-> " + self.product if self.product else "")


class NullProcess(Process):
    """ This is a null process with a 0 cross section it is useful 
    when we reduce other processes. """
    def __init__(self, target, kind):
        self.data = np.empty((0, 2))
        self.interp = lambda x: np.zeros_like(x)
        self.target_name = target
        self.kind = kind
        self.isnull = True

        self.comment = None
        self.mass_ratio = None
        self.product = None
        self.threshold = None
        self.weight_ratio = None

        self.x = np.array([])
        self.y = np.array([])

    def __str__(self):
        return "{NULL}"


def padinterp(data):
    """ Interpolates from data but adds elements at the beginning and end
    to extrapolate cross-sections. """
    if data[0, 0] > 0:
        x = np.r_[0.0, data[:, 0], 1e8]
        y = np.r_[data[0, 1], data[:, 1], data[-1, 1]]
    else:
        x = np.r_[data[:, 0], 1e8]
        y = np.r_[data[:, 1], data[-1, 1]]

    return interp1d(x, y, kind='linear')


def int_linexp0(a, b, u0, u1, g, x0):
    """ This is the integral in [a, b] of u(x) * exp(g * (x0 - x)) * x 
    assuming that
    u is linear with u({a, b}) = {u0, u1}."""

    # Since u(x) is linear, we calculate separately the coefficients
    # of degree 0 and 1 which, after multiplying by the x in the integrand
    # correspond to 1 and 2

    # The expressions involve the following exponentials that are problematic:
    # expa = np.exp(g * (-a + x0))
    # expb = np.exp(g * (-b + x0))
    # The problems come with small g: in that case, the exp() rounds to 1
    # and neglects the order 1 and 2 terms that are required to cancel the
    # 1/g**2 and 1/g**3 below.  The solution is to rewrite the expressions
    # as functions of expm1(x) = exp(x) - 1, which is guaranteed to be accurate
    # even for small x.
    expm1a = np.expm1(g * (-a + x0))
    expm1b = np.expm1(g * (-b + x0))

    ag = a * g
    bg = b * g

    ag1 = ag + 1
    bg1 = bg + 1

    g2 = g * g
    g3 = g2 * g

    # These are the expressions as functions of expa/expb
    # A1 = (  expa * ag1
    #        - expb * bg1) / g2

    # A2 = (expa * (2 * ag1 + ag * ag) - 
    #       expb * (2 * bg1 + bg * bg)) / g3

    A1 = (  expm1a * ag1 + ag
          - expm1b * bg1 - bg) / g2

    A2 = (expm1a * (2 * ag1 + ag * ag) + ag * (ag + 2) - 
          expm1b * (2 * bg1 + bg * bg) - bg * (bg + 2)) / g3

    # The factors multiplying each coefficient can be obtained by
    # the interpolation formula of u(x) = c0 + c1 * x
    c0 = (a * u1 - b * u0) / (a - b)
    c1 = (u0 - u1) / (a - b)

    r = c0 * A1 + c1 * A2

    # Where either F0 or F1 is 0 we return 0
    return np.where(np.isnan(r), 0.0, r)


