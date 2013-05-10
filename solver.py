""" These are the routines to actually solve the Boltzmann's equation.
    Here we are trying to follow as closely as possible the procedure
    described by Hagelaar and Pitchford (2005), henceforth referred
    as H&P. """

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
from scipy.sparse import dia_matrix, lil_matrix
from scipy.optimize import fsolve, fmin_l_bfgs_b

from process import Process
from target import Target, CombinedTarget

GAMMA = sqrt(2 * co.elementary_charge / co.electron_mass)
TOWNSEND = 1e-21


class BoltzmannSolver(object):
    def __init__(self, max_energy, n=1000):
        self.density = dict()
        self.max_energy = max_energy
        self.n = n
        
        self.EN = None

        # These are cell boundary values at i - 1/2
        self.benergy = np.linspace(0, max_energy, n + 1)

        # these are cell centers
        self.cenergy = 0.5 * (self.benergy[1:] + self.benergy[:-1])

        # And these are the deltas
        self.denergy = np.diff(self.benergy)

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
            

    def combine_all_targets(self):
        """ Combine all targets to create a mega-target that includes all
        collisions. """
        self.total = reduce(CombinedTarget, 
                            (t for t in self.target.values() if t.density > 0))



    def new_process_from_dict(self, d):
        """ Adds a new process from a dictionary with its properties. """
        proc = Process(**d)
        try:
            target = self.target[proc.target_name]
        except KeyError:
            target = Target(proc.target_name)
            self.target[proc.target_name] = target

        target.add_process(proc)

    
    def init(self):
        """ Does all the work previous to the actual iterations.
        The densities must be set at this point. """
        self.combine_all_targets()
        self.total.set_energy_grid(self.cenergy)

        self.sigma_eps = self.total.all_weighted_elastic.interp(self.benergy)
        self.sigma_m = self.total.all_all.interp(self.benergy)
        self.W = -GAMMA * self.benergy**2 * self.sigma_eps
        
        # This is the coeff of sigma_tilde
        self.DA = (GAMMA / 3. * self.EN**2 * self.benergy)

        # This is the independent term
        self.DB = (GAMMA * self.kT * self.benergy**2 * self.sigma_eps
                   / co.elementary_charge)

        logging.info("Solver succesfully initialized/updated")


    ##
    # Here are the functions that depend on F0 and are therefore
    # called in each iteration.  These are all pure-functions without
    # side-effects and without changing the state of self
    def maxwell(self, kT):
        return (2 * np.sqrt(self.cenergy / np.pi)
                * kT**(-3./2.) * np.exp(-self.cenergy / kT))

    def iterate(self, f0):
        def residual(f):
            A, Q = self.linsystem(f)
            r = A.dot(f) - Q.dot(f)
            logging.info("New linsystem calculated (delta=%g)"
                         % (sum(r**2)))
            return sum(r**2)

        bounds = [(0, None) for _ in f0]
        f1 = fmin_l_bfgs_b(residual, f0, bounds=bounds, approx_grad=True,
                           pgtol=1e-50)
        return f1


    def linsystem(self, F):
        F_func = self.F_as_func(F)
        nu_bar = self.nu_bar(F_func)
        sigma_tilde = self.sigma_tilde(F_func)
        A = self.scharf_gummel(F, sigma_tilde)

        if np.any(np.isnan(A.todense())):
            raise ValueError("NaN found in A")

        Q = self.PQ(F_func)
        if np.any(np.isnan(Q.todense())):
            raise ValueError("NaN found in Q")

        return A, Q


    def F_as_func(self, F):
        """ Builds a log interpolator for F. """
        logF = np.log(F)
        logF_func = interp1d(self.cenergy, logF, 
                             kind='linear', bounds_error=False,
                             fill_value=-np.inf)
        def F_func(x):
            return np.exp(logF_func(x))

        return F_func


    def nu_bar(self, F0_func):
        """ Calculates nu_bar/N/GAMMA from F0. """
        return (self.total.all_ionization.int_exp(F0_func)
                - self.total.all_attachment.int_exp(F0_func))


    def scharf_gummel(self, F0, sigma_tilde):
        D = self.DA / sigma_tilde + self.DB

        # Assuming here that the grid is homogeneous
        z = self.W * self.denergy[0] / D
        a0 = self.W / (1 - np.exp(-z))
        a1 = self.W / (1 - np.exp(z))


        #A = lil_matrix((self.n, self.n))
        diags = np.empty((3, self.n))
        diags[0, :] = a0[:-1] - a1[1:]
        diags[1, :] = a1[:-1]
        diags[2, :] = a0[1:]

        A = dia_matrix((self.n, self.n), (diags, [0, 1, -1]))
        return A


    def sigma_tilde(self, F0_func):
        nu = self.nu_bar(F0_func)
        return self.sigma_m + nu / np.sqrt(self.benergy)


    def PQ(self, F0_func):
        P = np.empty_like(self.cenergy)
        Q = lil_matrix((self.n, self.n))

        for i in xrange(self.n):
            Q[i, i] += self.total.all_inelastic.int_exp(
                F0_func, interval=[self.benergy[i], self.benergy[i + 1]])

            for k in self.total.inelastic:
                for j in xrange(i - int(np.ceil(k.threshold 
                                                / self.denergy[0])), 
                                i + 1):
                    eps1 = min(max(self.benergy[i] + k.threshold,
                                   self.benergy[j]),
                               self.benergy[i + 1])
                    eps2 = min(max(self.benergy[i + 1] + k.threshold,
                                   self.benergy[j]),
                               self.benergy[i + 1])

                    if eps1 == eps2:
                        continue
                    Q[i, j] += k.int_exp(F0_func, interval=[eps1, eps2])

        return Q.tocsr()
