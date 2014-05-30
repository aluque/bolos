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
from scipy import integrate
from scipy.interpolate import interp1d
from scipy import sparse
from scipy.optimize import fsolve, fmin_l_bfgs_b
from scipy.linalg import inv

from process import Process
from target import Target

GAMMA = sqrt(2 * co.elementary_charge / co.electron_mass)
TOWNSEND = 1e-21

class ConvergenceError(Exception):
    pass

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

        # This is useful when integrating the growth term.
        self.denergy32 = self.benergy[1:]**1.5 - self.benergy[:-1]**1.5

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

    def search(self, signature, product=None, first=True):
        if product is not None:
            l = self.target[signature].by_product[product]
            if not l:
                raise KeyError("Process %s not found" % signature)

            return l[0] if first else l

        t, p = [x.strip() for x in signature.split('->')]
        return self.search(t, p, first=first)


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

    def iter_growth(self):
        """ Iterates over all processes that affect the growth
        of electron density, i.e. ionization and attachment."""
        for target in self.target.values():
            if target.density > 0:
                for process in target.ionization:
                    yield target, process

                for process in target.attachment:
                    yield target, process

    def iter_momentum(self):
        """ Iterates over all processes."""
        for t, k in self.iter_elastic():
            yield t, k

        for t, k in self.iter_inelastic():
            yield t, k


    def init(self):
        """ Does all the work previous to the actual iterations.
        The densities must be set at this point. """

        self.sigma_eps = np.zeros_like(self.benergy)
        self.sigma_m = np.zeros_like(self.benergy)
        for target, process in self.iter_elastic():
            s = target.density * process.interp(self.benergy)
            self.sigma_eps += 2 * target.mass_ratio * s
            self.sigma_m += s
            process.set_grid_cache(self.grid)

        for target, process in self.iter_inelastic():
            self.sigma_m += target.density * process.interp(self.benergy)
            process.set_grid_cache(self.grid)

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


    def iterate(self, f0, delta=1e14):
        A, Q = self.linsystem(f0)

        f1 = sparse.linalg.spsolve(sparse.eye(self.n) 
                                   + delta * A - delta * Q, f0)

        return self.normalized(f1)

    
    def converge(self, f0, maxn=100, rtol=1e-5, delta0=1e14,
                 full=False, **kwargs):
        """ Iterates the attempted solution f0 until convergence is reached or
        maxn iterations are consumed.  """

        err0 = err1 = 0
        delta = delta0

        # This is just something that seems to work; can be optimized
        # to improve convergence.
        m = 4

        for i in xrange(maxn):
            # If we have already two error estimations we use Richardson
            # extrapolation to obtain a new delta and speed up convergence.
            if 0 < err1 < err0:
                # Linear extrapolation
                # delta = delta * err1 / (err0 - err1)

                # Log extrapolation attempting to reduce the error a factor m
                delta = delta * np.log(m) / (np.log(err0) - np.log(err1))
                
            f1 = self.iterate(f0, delta=delta, **kwargs)
            err0 = err1
            err1 = self.norm(abs(f0 - f1))
            
            logging.debug("After iteration %3d, err = %g (target: %g)" 
                          % (i + 1, err1, rtol))
            if err1 < rtol:
                logging.info("Convergence achieved after %d iterations. "
                             "err = %g" % (i + 1, err1))
                if full:
                    return f1, i + 1, err1

                return f1
            f0 = f1
            
        logging.error("Convergence failed")

        raise ConvergenceError()


    def linsystem(self, F):
        Q = self.PQ(F)

        # if np.any(np.isnan(Q.todense())):
        #     raise ValueError("NaN found in Q")

        nu = np.sum(Q.dot(F))

        sigma_tilde = self.sigma_m + nu / np.sqrt(self.benergy) / GAMMA

        # The R (G) term, which we add to A.
        G = 2 * self.denergy32 * nu / 3

        A = self.scharf_gummel(sigma_tilde, G)

        # if np.any(np.isnan(A.todense())):
        #     raise ValueError("NaN found in A")

        return A, Q


    def norm(self, f):
        return integrate.simps(f * np.sqrt(self.cenergy), x=self.cenergy)
        
        # return np.sum(f * np.sqrt(self.cenergy) * self.denergy)

    def normalized(self, f):
        N = self.norm(f)
        return f / N


    def scharf_gummel(self, sigma_tilde, G=0):
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

        diags[0, :] += G

        A = sparse.dia_matrix((diags, [0, 1, -1]), shape=(self.n, self.n))

        return A


    def g(self, F0):
        Fp = np.r_[F0[0], F0, F0[-1]]
        cenergyp = np.r_[self.cenergy[0], self.cenergy, self.cenergy[-1]]
        g = np.log(Fp[2:] / Fp[:-2]) / (cenergyp[2:] - cenergyp[:-2])
        
        return g


    def PQ(self, F0, reactions=None):
        PQ = sparse.csr_matrix((self.n, self.n))

        g = self.g(F0)
        if reactions is None:
            reactions = list(self.iter_inelastic())

        for t, k in reactions:
            r = t.density * GAMMA * k.scatterings(g, self.cenergy)
            in_factor = k.in_factor
            
            Q = sparse.coo_matrix((in_factor * r, (k.i, k.j)),
                                   shape=(self.n, self.n))
            P = sparse.coo_matrix((-r, (k.j, k.j)),
                                  shape=(self.n, self.n))


            PQ = PQ + Q.tocsr() + P.tocsr()


        return PQ

        
    def rate(self, F0, k, weighted=False):
        g = self.g(F0)
        r = k.scatterings(g, self.cenergy)

        P = sparse.coo_matrix((GAMMA * r, (k.j, np.zeros(r.shape))), 
                              shape=(self.n, 1)).todense()

                              
        r = F0.dot(P)
        if weighted:
            r *= k.target.density

        return r


    ##
    # Now some functions to calculate transport parameters from the
    # converged F0
    def mobility(self, F0):
        """ Calculates the mobility * N from a converged distribution function.
        """

        DF0 = np.r_[0.0, np.diff(F0) / np.diff(self.cenergy), 0.0]
        Q = self.PQ(F0, reactions=self.iter_growth())

        nu = np.sum(Q.dot(F0)) / GAMMA
        sigma_tilde = self.sigma_m + nu / np.sqrt(self.benergy)

        y = DF0 * self.benergy / sigma_tilde
        y[0] = 0

        return -(GAMMA / 3) * integrate.simps(y, x=self.benergy)


    def diffusion(self, F0):
        """ Calculates the mobility * N from a converged distribution function.
        """

        Q = self.PQ(F0, reactions=self.iter_growth())

        nu = np.sum(Q.dot(F0)) / GAMMA

        sigma_m = np.zeros_like(self.cenergy)
        for target, process in self.iter_momentum():
            s = target.density * process.interp(self.cenergy)
            sigma_m += s

        sigma_tilde = sigma_m + nu / np.sqrt(self.cenergy)

        y = F0 * self.cenergy / sigma_tilde

        return (GAMMA / 3) * integrate.simps(y, x=self.cenergy)
