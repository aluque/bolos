""" These are the routines to actually solve the Boltzmann's equation.
    Here we are trying to follow as closely as possible the procedure
    described by Hagelaar and Pitchford (2005), henceforth referred
    as H&P. """

import sys
import logging

from math import sqrt
import numpy as np

# Units in this module will be SI units, except energies, which are expressed
# in eV.
# The scipy.constants contains the recommended CODATA for all physical
# constants in SI units.
import scipy.constants as co
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from scipy import sparse
from scipy.optimize import fsolve, fmin_l_bfgs_b
from scipy.linalg import inv

from process import Process
from target import Target

# For debugging only: remove later
import pylab

from IPython import embed

GAMMA = sqrt(2 * co.elementary_charge / co.electron_mass)
TOWNSEND = 1e-21


class BoltzmannSolver(object):
    def __init__(self, grid):
        self.density = dict()
        self.n = grid.n
        
        self.EN = None

        self.grid = grid

        # These are cell boundary values at i - 1/2
        self.benergy = self.grid.b

        # these are cell centers
        self.cenergy = self.grid.c

        # And these are the deltas
        self.denergy = self.grid.d

        # A dictionary with target_name -> target
        self.target = {}
        

    def load_collisions(self, dict_processes):
        """ Loads the set of collisions from the list of processes. """
        for p in dict_processes:
            self.new_process_from_dict(p)

        # We make sure that all targets have their elastic cross-sections
        # in the form of ELASTIC cross sections (not EFFECTIVE / MOMENTUM)
        for key, item in self.target.iteritems():
            item.ensure_elastic()
            

    def new_process_from_dict(self, d):
        """ Adds a new process from a dictionary with its properties. """
        proc = Process(**d)
        try:
            target = self.target[proc.target_name]
        except KeyError:
            target = Target(proc.target_name)
            self.target[proc.target_name] = target

        target.add_process(proc)


    def iter_elastic(self):
        """ Iterates over all elastic processes yielding (target, process)
        tuples. """
        for target in self.target.values():
            if target.density > 0:
                for process in target.elastic:
                    yield target, process

    def iter_inelastic(self):
        """ Iterates over all inelastic processes yielding (target, process)
        tuples. """
        for target in self.target.values():
            if target.density > 0:
                for process in target.inelastic:
                    yield target, process


    def init(self):
        """ Does all the work previous to the actual iterations.
        The densities must be set at this point. """

        self.sigma_eps = np.zeros_like(self.benergy)
        self.sigma_m = np.zeros_like(self.benergy)
        for target, process in self.iter_elastic():
            s = target.density * process.interp(self.benergy)
            self.sigma_eps += 2 * target.mass_ratio * s
            self.sigma_m += s

        for target, process in self.iter_inelastic():
            self.sigma_m += target.density * process.interp(self.benergy)


        self.W = -GAMMA * self.benergy**2 * self.sigma_eps
        
        # This is the coeff of sigma_tilde
        self.DA = (GAMMA / 3. * self.EN**2 * self.benergy)

        # This is the independent term
        self.DB = (GAMMA * self.kT * self.benergy**2 * self.sigma_eps)

        logging.info("Solver succesfully initialized/updated")


    ##
    # Here are the functions that depend on F0 and are therefore
    # called in each iteration.  These are all pure-functions without
    # side-effects and without changing the state of self
    def maxwell(self, kT):
        return (2 * np.sqrt(1 / np.pi)
                * kT**(-3./2.) * np.exp(-self.cenergy / kT))


    def iterate(self, f0, eps=1e15):
        A, Q = self.linsystem(f0)

        f1 = sparse.linalg.spsolve(sparse.eye(self.n) + eps * A - eps * Q, f0)
        return self.normalized(f1)

    
    def converge(self, f0, maxn=100, rtol=1e-5, **kwargs):
        """ Iterates the attempted solution f0 until convergence is reached or
        maxn iterations are consumed.  """
        from matplotlib import cm
        pylab.figure(1)

        for i in xrange(maxn):
            pylab.plot(self.cenergy, f0, lw=1.8, c=cm.jet(float(i) / maxn))

            f1 = self.iterate(f0, **kwargs)
            err = self.norm(abs(f0 - f1))
            
            logging.debug("After iteration %3d, err = %g (target: %g)" 
                          % (i + 1, err, rtol))
            if err < rtol:
                logging.info("Convergence achieved after %d iterations. "
                             "err = %g" % (i + 1, err))
                return f1
            f0 = f1
            
        logging.error("Convergence failed")


    def linsystem(self, F):
        Q = self.PQ(F)

        if np.any(np.isnan(Q.todense())):
            raise ValueError("NaN found in Q")

        nu = np.sum(Q.dot(F)) / GAMMA

        logging.debug("Growth factor nu = %g" % (GAMMA * nu))
        sigma_tilde = self.sigma_m + nu / np.sqrt(self.benergy)

        A = self.scharf_gummel(sigma_tilde)
        
        if np.any(np.isnan(A.todense())):
            raise ValueError("NaN found in A")

        return A, Q


    def norm(self, f):
        return np.sum(f * np.sqrt(self.cenergy) * self.denergy)


    def normalized(self, f):
        return f / self.norm(f)


    def scharf_gummel(self, sigma_tilde):
        D = self.DA / (sigma_tilde) + self.DB

        # Due to the zero flux b.c. the values of z[0] and z[-1] are never used.
        # To make sure, we set is a nan so it will taint everything if ever 
        # used.
        # TODO: Perhaps it would be easier simply to set the appropriate
        # values here to satisfy the b.c.
        z  = self.W * np.r_[np.nan, np.diff(self.cenergy), np.nan] / D

        a0 = self.W / (1 - np.exp(-z))
        a1 = self.W / (1 - np.exp(z))

        diags = np.zeros((3, self.n))

        # No flux at the energy = 0 boundary
        diags[0, 0]  = a0[1]

        diags[0, 1:] =  a0[2:]  - a1[1:-1]
        diags[1, :]  =  a1[:-1]
        diags[2, :]  = -a0[1:]

        # F[n+1] = 2 * F[n] + F[n-1] b.c.
        # diags[2, -2] -= a1[-1]
        # diags[0, -1] += 2 * a1[-1]

        # F[n+1] = F[n] b.c.
        # diags[0, -1] += a1[-1]

        # zero flux b.c.
        diags[2, -2] = -a0[-2]
        diags[0, -1] = -a1[-2]

        A = sparse.dia_matrix((diags, [0, 1, -1]), shape=(self.n, self.n))

        return A


    def g(self, F0):
        Fp = np.r_[F0[0], F0, F0[-1]]
        cenergyp = np.r_[self.cenergy[0], self.cenergy, self.cenergy[-1]]
        g = np.log(Fp[2:] / Fp[:-2]) / (cenergyp[2:] - cenergyp[:-2])
        
        return g


    def PQ(self, F0):
        Q = sparse.lil_matrix((self.n, self.n))
        g = self.g(F0)

        for i in xrange(self.n):
            for t, k in self.iter_inelastic():
                in_factor = k.in_factor

                # This is the the range of energies where a collision
                # would add a particle to cell i.
                # The 1e-9 appears in eps_b to make sure that it is contained
                # in the open interval (0, self.benergy[-1])
                eps_a = k.shift_factor * self.benergy[i] + k.threshold
                eps_b = min(k.shift_factor * self.benergy[i + 1] + k.threshold,
                            self.benergy[-1] - 1e-9)

                # And these are the cells where these energies are located
                ja = self.grid.cell(eps_a)
                jb = self.grid.cell(eps_b)

                for j in xrange(ja, jb + 1):
                    eps1 = max(eps_a, self.benergy[j])
                    eps2 = min(eps_b, self.benergy[j + 1])

                    if eps1 > eps2:
                        continue

                    r = t.density * GAMMA * k.int_exp0(g[j], self.cenergy[j],
                                                       interval=[eps1, eps2])
                        
                    Q[i, j] += in_factor * r 
                    Q[j, j] -= r


        return Q.tocsr()
