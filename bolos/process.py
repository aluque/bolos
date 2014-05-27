import logging

import numpy as np
from scipy.interpolate import interp1d


class Process(object):
    # The factor of in-scatering.  It should never be used for elastic
    # collision, so we set it as None to trigger an exception if it is ever
    # used.
    IN_FACTOR = {'EXCITATION': 1,
                 'IONIZATION': 2,
                 'ATTACHMENT': 0,
                 'ELASTIC': None,
                 'MOMENTUM': None}

    # The shift factor for inelastic collisions. 
    SHIFT_FACTOR = {'EXCITATION': 1,
                    'IONIZATION': 2,
                    'ATTACHMENT': 1,
                    'ELASTIC': None,
                    'MOMENTUM': None}

                 
    def __init__(self, target=None, kind=None, data=None,
                 comment='', mass_ratio=None,
                 product=None, threshold=None, weight_ratio=None):
        self.target_name = target

        # We will link this later
        self.target = None

        self.kind = kind
        self.data = np.array(data)
        # Normalize to forget roundoff errors
        self.data[:] = np.where(np.abs(data) < 1e-35, 0.0, data)

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


    def int_exp0(self, g, epsj, interval=None):
        """ Integrates sigma * eps * exp(g (epsj - eps)) in the given interval.
        See below, int_linexp0 for the shape that we assume for linexp. """
        if interval is None:
            interval = [self.x[0], self.x[-1]]

        inflt = np.logical_and(self.x > interval[0], self.x < interval[1])
        x = np.r_[interval[0], self.x[inflt], interval[1]]
        sigma = np.r_[[self.interp(interval[0])], self.y[inflt], 
                      [self.interp(interval[1])]]
        
        return np.sum(int_linexp0(x[:-1], x[1:], 
                                  sigma[:-1], sigma[1:], 
                                  g, epsj))


    def int_expij(self, i, j, g, epsj):
        """ As int_exp0 but uses the cache of cell intersections. """

        x = self.xij[(i, j)]
        sigma = self.sigmaij[(i, j)]

        return np.sum(int_linexp0(x[:-1], x[1:], 
                                  sigma[:-1], sigma[1:], 
                                  g, epsj))


    def set_grid_cache(self, grid):
        """ Sets a grid cache of the intersections between grid cell j and grid
        cell i shifted. """

        if self.cached_grid is grid:
            # We only have to redo all these computations when the grid changes
            # so we store the grid for which this has been already calculated.
            return


        self.sigmaij = {}
        self.xij = {}
        self.ja = np.zeros((grid.n), dtype='i')
        self.jb = np.zeros((grid.n), dtype='i')

        for i in xrange(grid.n):
            # This is the the range of energies where a collision
            # would add a particle to cell i.
            # The 1e-9 appears in eps_b to make sure that it is contained
            # in the open interval (0, self.benergy[-1])
            eps_a = self.shift_factor * grid.b[i] + self.threshold
            eps_b = min(self.shift_factor * grid.b[i + 1] 
                        + self.threshold,
                        grid.b[-1] - 1e-9)

            # And these are the cells where these energies are located
            self.ja[i] = grid.cell(eps_a)
            self.jb[i] = grid.cell(eps_b)

            for j in xrange(self.ja[i], self.jb[i] + 1):
                eps1 = max(eps_a, grid.b[j])
                eps2 = min(eps_b, grid.b[j + 1])
                yeps1 = self.interp(eps1)
                yeps2 = self.interp(eps2)

                inflt = np.logical_and(self.x > eps1, 
                                       self.x < eps2)

                self.xij[(i, j)] = np.r_[eps1, self.x[inflt], eps2]
                self.sigmaij[(i, j)] = np.r_[[yeps1], self.y[inflt], [yeps2]]

                

            self.cached_grid = grid


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

    def int_exp0(self, g, F, interval=None):
        return 0

    def __str__(self):
        return "{NULL}"


def padinterp(data):
    """ Interpolates from data but adds elements at the beginning and end
    to extrapolate cross-sections. """
    if data[0, 0] > 0:
        x = np.r_[0.0, data[:, 0], 1e8]
        y = np.r_[0.0, data[:, 1], data[-1, 1]]
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
    expa = np.exp(g * (-a + x0))
    expb = np.exp(g * (-b + x0))

    ag = a * g
    bg = b * g

    ag1 = ag + 1
    bg1 = bg + 1

    g2 = g * g
    g3 = g2 * g

    A1 = (  expa * ag1
          - expb * bg1) / g2

    A2 = (expa * (2 * ag1 + ag * ag) - 
          expb * (2 * bg1 + bg * bg)) / g3

    # The factors multiplying each coefficient can be obtained by
    # the interpolation formula of u(x) = c0 + c1 * x
    c0 = (a * u1 - b * u0) / (a - b)
    c1 = (u0 - u1) / (a - b)

    r = c0 * A1 + c1 * A2

    # Where either F0 or F1 is 0 we return 0
    return np.where(np.isnan(r), 0.0, r)


