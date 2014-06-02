import logging

import numpy as np
from scipy.interpolate import interp1d


class Process(object):
    # The factor of in-scatering.  
    IN_FACTOR = {'EXCITATION': 1,
                 'IONIZATION': 2,
                 'ATTACHMENT': 0,
                 'ELASTIC': 1,
                 'MOMENTUM': 1}

    # The shift factor for inelastic collisions. 
    SHIFT_FACTOR = {'EXCITATION': 1,
                    'IONIZATION': 2,
                    'ATTACHMENT': 1,
                    'ELASTIC': 1,
                    'MOMENTUM': 1}

                 
    def __init__(self, target=None, kind=None, data=None,
                 comment='', mass_ratio=None,
                 product=None, threshold=0, weight_ratio=None):
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


    def scatterings(self, g, eps):
        # if len(self.j) == 0:
        #     # When we do not have inelastic collisions or when the grid is
        #     # smaller than the thresholds, we still return an empty array
        #     # and thus avoid exceptions in g[self.j]
        #     return np.array([], dtype='f')

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

        self.i, self.j, self.eps, self.sigma = [], [], [], []

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
            ja = grid.cell(eps_a)
            jb = grid.cell(eps_b)


            for j in xrange(ja, jb + 1):
                eps1 = max(eps_a, grid.b[j])
                eps2 = min(eps_b, grid.b[j + 1])
                yeps1 = self.interp(eps1)
                yeps2 = self.interp(eps2)

                inflt = np.logical_and(self.x > eps1, 
                                       self.x < eps2)
                
                xij = np.r_[eps1, self.x[inflt], eps2]
                sigmaij = np.r_[[yeps1], self.y[inflt], [yeps2]]

                for k in xrange(len(xij) - 1):
                    self.i.append(i)
                    self.j.append(j)
                    self.eps.append([xij[k], xij[k + 1]])
                    self.sigma.append([sigmaij[k], sigmaij[k + 1]])


        self.i = np.array(self.i)
        self.j = np.array(self.j)
        self.eps = np.array(self.eps)
        self.sigma = np.array(self.sigma)

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


