import logging

import numpy as np
from scipy.interpolate import interp1d


class Process(object):
    def __init__(self, target=None, kind=None, data=None,
                 comment='', mass_ratio=None,
                 product=None, threshold=None, weight_ratio=None):
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

        if np.amin(self.data[:, 0]) < 0:
            raise ValueError("Negative energy in the cross section %s"
                             % str(self))
 
        if np.amin(self.data[:, 1]) < 0:
            raise ValueError("Negative cross section for %s"
                             % str(self))
       

    def set_energy_grid(self, eps):
        """ Performs all computations needed for an energy grid;
        this is to avoid repeated operations during the iterations. """
        self.eps = eps
        xeps = np.r_[self.x, eps]
        yeps = np.r_[self.y, self.interp(eps)]

        isort = np.argsort(xeps)
        self.xeps = xeps[isort]
        self.yeps = yeps[isort]

        logging.debug("Energy grid set in %s" % str(self))


    def int_exp(self, F, interval=None):
        """ Integrates sigma * eps * F in the given interval.
        See below, int_linexp for the shape that we assume for linexp. """
        if interval is None:
            interval = [self.x[0], self.x[-1]]

        # logging.debug("interval = %s" % repr(interval))

        inflt = np.logical_and(self.xeps > interval[0], self.xeps < interval[1])
        x = np.r_[interval[0], self.xeps[inflt], interval[1]]
        
        sigma = np.r_[[self.interp(interval[0])], self.yeps[inflt], 
                      [self.interp(interval[1])]]
        Fx = F(x)

        # logging.debug("x = %s" % repr(x))
        # logging.debug("sigma = %s" % repr(sigma))
        # logging.debug("Fx = %s" % repr(Fx))

        return np.sum(int_linexp(x[:-1], x[1:], 
                                 sigma[:-1], sigma[1:], 
                                 Fx[:-1], Fx[1:]))

                          

    def plot(self, ax, *args, **kwargs):
        """ Plots the cross sections of this process into ax.  All kwargs
        are passed to pylab's plot. """
        ax.plot(self.data[:, 0], self.data[:, 1], *args, **kwargs)


    def __str__(self):
        return "{%s: %s %s}" % (self.kind, self.target_name, 
                                "-> " + self.product if self.product else "")


class ScaledProcess(Process):
    def __init__(self, orig, factor):
        """ Returns a new process with the cross section scaled by a factor
        factor.  Generally, factor will be a molar fraction. """

        newdata = orig.data.copy()
        newdata[:, 1] *= factor
        self.orig = orig
        self.factor = factor

        super(ScaledProcess, self).__init__(target=orig.target_name, 
                                            kind=orig.kind, 
                                            data=newdata,
                                            comment=orig.comment, 
                                            mass_ratio=orig.mass_ratio,
                                            product=orig.product, 
                                            threshold=orig.threshold, 
                                            weight_ratio=orig.weight_ratio)

    def __str__(self):
        return "{%s * %s}" % (str(self.factor), str(self.orig))


class CombinedProcess(Process):
    def __init__(self, he, she):
        """ Initializes a process that combines this one with other.
        The cross sections are interpolated, and then added. """

        his_x = he.data[:, 0].copy()
        his_y = he.data[:, 1].copy()

        her_x = she.data[:, 0].copy()
        her_y = she.data[:, 1].copy()
        
        # We have to avoid data repetition here.
        only_hers = np.logical_not(np.in1d(her_x, his_x))
        her_x = her_x[only_hers]
        her_y = her_y[only_hers]

        his_y = his_y + she.interp(his_x)
        her_y = her_y + he.interp(her_x)

        our_x = np.r_[his_x, her_x]
        our_y = np.r_[his_y, her_y]

        isort = np.argsort(our_x)
        
        data = np.c_[our_x[isort], our_y[isort]]
        target = ' '.join({he.target_name, she.target_name})
        kind = ' '.join({he.kind, she.kind})

        self.he = he
        self.she = she

        # Many properties here stop making sense now, so we set them as None
        super(CombinedProcess, self).__init__(target=target, 
                                              kind=kind, 
                                              data=data,
                                              comment='', 
                                              mass_ratio=None,
                                              product=None, 
                                              threshold=None, 
                                              weight_ratio=None)


    def __str__(self):
        return "{%s + %s}" % (str(self.he), str(self.she))



class NullProcess(Process):
    """ This is a null process with a 0 cross section it is useful 
    when we reduce other processes. """
    def __init__(self, target, kind):
        self.data = np.empty((0, 2))
        self.interp = lambda x: 0.0
        self.target_name = target
        self.kind = kind
        self.isnull = True

        self.comment = None
        self.mass_ratio = None
        self.product = None
        self.threshold = None
        self.weight_ratio = None


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

import numexpr

def int_linexp(a, b, u0, u1, F0, F1):
    """ This is the integral in [a, b] of u(x) * F(x) * x assuming that
    u is linear with u({a, b}) = {u0, u1} and F is exponential
    with F({a, b}) = {F0, F1}. """
    ab = a - b
    logF1F0 = np.log(F1 / F0)
    

    r = ((ab*(2*ab*(F0 - F1)*(u0 - u1) + 
              logF1F0*(2*a*F0*u0 - b*(F0 + F1)*u0 + 2*b*F1*u1 - 
                       a*(F0 + F1)*u1 + (a*F0*u0 - b*F1*u1)*logF1F0)))/
         logF1F0**3)

    # Where either F0 or F1 is 0 we return 0
    return np.where(np.isnan(r), 0.0, r)
