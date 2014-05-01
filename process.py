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

    # The shift factor for inelastic collisions. Again, we set it as None for
    # elastic collisions.
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
       

    def int_exp0(self, g, epsj, interval=None):
        """ Integrates sigma * eps * exp(g (epsj - eps)) in the given interval.
        See below, int_linexp0 for the shape that we assume for linexp. """
        if interval is None:
            interval = [self.x[0], self.x[-1]]

        # inflt = np.logical_and(self.xeps > interval[0], self.xeps < interval[1])
        # x = np.r_[interval[0], self.xeps[inflt], interval[1]]
        # sigma = np.r_[[self.interp(interval[0])], self.yeps[inflt], 
        #               [self.interp(interval[1])]]

        inflt = np.logical_and(self.x > interval[0], self.x < interval[1])
        x = np.r_[interval[0], self.x[inflt], interval[1]]
        sigma = np.r_[[self.interp(interval[0])], self.y[inflt], 
                      [self.interp(interval[1])]]
        
        return np.sum(int_linexp0(x[:-1], x[1:], 
                                  sigma[:-1], sigma[1:], 
                                  g, epsj))



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
    A1 = (  np.exp(g * (-a + x0)) * (1 + a * g) 
          - np.exp(g * (-b + x0)) * (1 + b * g)) / g**2

    A2 = (np.exp(g * (-a + x0)) * (2 + a * g * (2 + a * g)) - 
          np.exp(g * (-b + x0)) * (2 + b * g * (2 + b * g))) / g**3

    # The factors multiplying each coefficient can be obtained by
    # the interpolation formula of u(x) = c0 + c1 * x
    c0 = (a * u1 - b * u0) / (a - b)
    c1 = (u0 - u1) / (a - b)

    r = c0 * A1 + c1 * A2

    # Where either F0 or F1 is 0 we return 0
    return np.where(np.isnan(r), 0.0, r)


