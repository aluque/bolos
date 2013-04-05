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
    def nu_bar(self, F0):
        """ Calculates nu_bar/N/GAMMA from F0. """
        return (self.total.all_ionization.int_exp(F0)
                - self.total.all_attachment.int_exp(F0))


    def sigma_tilde(self, F0):
        nu = self.nu_bar(F0)
        return self.sigma_m + nu / sqrt(self.benergy)


    def P(self, F0):
        pass
    
