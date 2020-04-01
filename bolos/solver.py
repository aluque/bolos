""" This module contains the main routines to load processes, specify the
physical conditions and solve the Boltzmann equation.
 
The data and calculations are encapsulated into the :class:`BoltzmannSolver` 
class, which you have to instantiate with a :class:`grid.Grid` instance.  
Use :func:`BoltzmannSolver.load_collisions` or 
:func:`BoltzmannSolver.add_process` to add processes with
their cross-sections.  Afterwards, set the density of each component
with :func:`BoltzmannSolver.set_density` or :attr:`BoltzmannSolver.target`.
The method :func:`BoltzmannSolver.maxwell` gives you a reasonable initial guess
for the electron energy distribution function (EEDF) that you can then improve
iteratively with :func:`BoltzmannSolver.converge`.  Finally, methods such as
:func:`BoltzmannSolver.rate` or :func:`BoltzmannSolver.mobility` allow you
to obtain reaction rates and transport parameters for a given EEDF.

"""
from __future__ import absolute_import

from builtins import range
from builtins import object
__docformat__ = "restructuredtext en"

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
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .process import Process
from .target import Target

GAMMA = sqrt(2 * co.elementary_charge / co.electron_mass)
TOWNSEND = 1e-21
KB = co.k
ELECTRONVOLT = co.eV

class ConvergenceError(Exception):
    pass

class BoltzmannSolver(object):
    """Class to solve the Boltzmann equation for electrons in a gas.

    This class contains the required elements to specify the conditions
    for the solver and obtain the equilibrium electron energy distribution
    function.

    Parameters
    ----------
    grid : :class:`grid.Grid`
       The grid in energies where the distribution funcition will be 
       evaluated.

    Attributes
    ----------
    benergy : array of floats
       Cell boundaries of the energy grid (set automatically at \
       initialization). Equivalent to `grid.b`.  
    benergy : array of floats
       Cell lengths of the energy grid (set automatically at initialization). \
       Equivalent to `grid.d`. 
    cenergy : array of floats
       Cell centers of the energy grid (set automatically at initialization). \
       Equivalent to `grid.c`.
    n : int 
       Number of cells in the energy grid (set automatically at \
       initialization). Equivalent to `grid.n`.
    kT : float
       Gas temperature in eV.  Must be set by the user.
    EN : float
       Reduced electric field in Townsend (1 Td is 1e-21 V m^2). \
       Must be set by the user.
    target : dict
       A dictionary with targets in the set of processes.\
       The user needs to set the density (molar fraction) of the desired \
       targets using this dictionary.  E.g. synthetic air is represented by 

    Examples
    --------
    >>> import numpy as np
    >>> from bolos import solver, grid
    >>> grid.LinearGrid(0, 60., 400)
    >>> bsolver = solver.BoltzmannSolver(grid)
    >>> # Parse the cross-section file in BOSIG+ format and load it into the
    >>> # solver.
    >>> with open(args.input) as fp:
    >>>     processes = parser.parse(fp)
    >>> bsolver.load_collisions(processes)
    >>> 
    >>> # Set the conditions.  And initialize the solver
    >>> bsolver.target['N2'].density = 0.8
    >>> bsolver.target['O2'].density = 0.2
    >>> bsolver.kT = 300 * co.k / co.eV
    >>> bsolver.EN = 300.0 * solver.TOWNSEND
    >>> bsolver.init()
    >>> 
    >>> # Start with Maxwell EEDF as initial guess.  Here we are starting with
    >>> # with an electron temperature of 2 eV
    >>> f0 = bsolver.maxwell(2.0)
    >>> 
    >>> # Solve the Boltzmann equation with a tolerance rtol and maxn 
    >>> # iterations.
    >>> f1 = bsolver.converge(f0, maxn=50, rtol=1e-5)

    """


    def __init__(self, grid):
        """ Initialize a solver instance.

        Use this method to initialize a solver instance with a given grid.

        Parameters
        ----------
        grid : :class:`grid.Grid`
               The grid in energies where the distribution funcition will be 
               evaluated.

        Returns
        -------
        """

        self.density = dict()
        
        self.EN = None

        self.grid = grid

        # A dictionary with target_name -> target
        self.target = {}

        # Default temporal growth model
        self.growth_model = 1

        # Default coulomb cross sections not taken into account
        self.coulomb = False
        
    def _get_grid(self):
        return self._grid

    def _set_grid(self, grid):
        self._grid = grid

        # These are cell boundary values at i - 1/2
        self.benergy = self.grid.b

        # these are cell centers
        self.cenergy = self.grid.c

        # And these are the deltas
        self.denergy = self.grid.d

        # This is useful when integrating the temporal growth term.
        self.denergy32 = self.benergy[1:]**1.5 - self.benergy[:-1]**1.5

        # This is useful when integrating the spatial growth term.
        self.denergy2 = self.benergy[1:]**2 - self.benergy[:-1]**2

        self.n = grid.n

    grid = property(_get_grid, _set_grid)


    def set_density(self, species, density):
        """ Sets the molar fraction of a species.

        Parameters
        ----------
        species : str
           The species whose density you want to set.
        density : float
           New value of the density.

        Returns
        -------

        Examples
        --------
        These are two equivalent ways to set densities for synthetic air:

        Using :func:`set_density`::

            bsolver.set_density('N2', 0.8)
            bsolver.set_density('O2', 0.2)
        
        Using `bsolver.target`::

            bsolver.target['N2'].density = 0.8
            bsolver.target['O2'].density = 0.2
        """

        self.target[species].density = density


    def load_collisions(self, dict_processes):
        """ Loads the set of collisions from the list of processes. 

        Loads a list of dictionaries containing processes.

        Parameters
        ----------
        dict_processes : List of dictionary or dictionary-like elements.
           The processes to add to this solver class.  
           See :method:`solver.add_process` for the required fields
           of each of the dictionaries.

        Returns
        -------
        processes : list
           A list of all added processes, as :class:`process.Process` instances.

        See Also
        --------
        add_process : Add a single process, with its cross-sections, to this
           solver.

        """
        plist = [self.add_process(**p) for p in dict_processes]

        # We make sure that all targets have their elastic cross-sections
        # in the form of ELASTIC cross sections (not EFFECTIVE / MOMENTUM)
        for key, item in self.target.items():
            item.ensure_elastic()

        return plist

    def add_process(self, **kwargs):
        """ Adds a new process to the solver.
        
        Adds a new process to the solver.  The process data is passed with
        keyword arguments.

        Parameters
        ----------
        type : string
           one of "EFFECTIVE", "MOMENTUM", "EXCITATION", "IONIZATION"
           or "ATTACHMENT".
        target : string
           the target species of the process (e.g. "O", "O2"...).
        ratio : float
           the ratio of the electron mass to the mass of the target
           (for elastic/momentum reactions only).
        threshold : float
           the energy threshold of the process in eV (only for 
           inelastic reactions).
        data : array or array-like
           cross-section of the process array with two columns: column
           0 must contain energies in eV, column 1 contains the
           cross-section in square meters for each of these energies.

        Returns
        -------
        process : :class:`process.Process`
           The process that has been added.

        Examples
        --------
        >>> import numpy as np
        >>> from bolos import solver, grid
        >>> grid.LinearGrid(0, 60., 400)
        >>> solver = BoltzmannSolver(grid)
        >>> # This is an example cross-section that decays exponentially
        >>> energy = np.linspace(0, 10)
        >>> cross_section = 1e-20 * np.exp(-energy)
        >>> solver.add_process(type="EXCITATION", target="Kriptonite", 
        >>>                    ratio=1e-5, threshold=10, 
        >>>                    data=np.c_[energy, cross_section])

        See Also
        --------
        load_collisions : Add a set of collisions.
        
        """
        proc = Process(**kwargs)
        try:
            target = self.target[proc.target_name]
        except KeyError:
            target = Target(proc.target_name)
            self.target[proc.target_name] = target

        target.add_process(proc)

        return proc


    def search(self, signature, product=None, first=True):
        """ Search for a process or a number of processes within the solver.
        
        Parameters
        ----------
        signature : string
           Signature of the process to search for.  It must be in the form
           "TARGET -> RESULT [+ RESULT2]...".     
        product : string
           If present, the first parameter is interpreted as TARGET and the
           second parameter is the PRODUCT.
        first : boolean
           If true returns only the first process matching the search; if
           false returns a list of them, even if there is only one result.
        
        Returns
        -------
        processes : list or :class:`process.Process` instance.
           If ``first`` was true, returns the first process matching the
           search.  Otherwise returns a (possibly empty) list of matches.

        Examples
        --------
        >>> ionization = solver.search("N2 -> N2^+")[0]
        >>> ionization = solver.search("N2", "N2^+", first=True)

        """
        if product is not None:
            l = self.target[signature].by_product[product]
            if not l:
                raise KeyError("Process %s not found" % signature)

            return l[0] if first else l

        t, p = [x.strip() for x in signature.split('->')]
        return self.search(t, p, first=first)


    def iter_elastic(self):
        """ Iterates over all elastic processes.

        Parameters
        ----------

        Returns
        -------
        An iterator over (target, process) tuples. 
        """

        for target in list(self.target.values()):
            if target.density > 0:
                for process in target.elastic:
                    yield target, process


    def iter_inelastic(self):
        """ Iterates over all inelastic processes.

        Parameters
        ----------

        Returns
        -------
        An iterator over (target, process) tuples. """

        for target in list(self.target.values()):
            if target.density > 0:
                for process in target.inelastic:
                    yield target, process


    def iter_growth(self):
        """ Iterates over all processes that affect the growth
        of electron density, i.e. ionization and attachment.

        Parameters
        ----------

        Returns
        -------
        An iterator over (target, process) tuples. 

        """
        for target in list(self.target.values()):
            if target.density > 0:
                for process in target.ionization:
                    yield target, process

                for process in target.attachment:
                    yield target, process

    def iter_all(self):
        """ Iterates over all processes.

        Parameters
        ----------

        Returns
        -------
        An iterator over (target, process) tuples. 

        """
        for t, k in self.iter_elastic():
            yield t, k

        for t, k in self.iter_inelastic():
            yield t, k


    def iter_momentum(self):
        return self.iter_all()


    def init(self):
        """ Initializes the solver with given conditions and densities of the
        target species.

        This method does all the work previous to the actual iterations.
        It has to be called whenever the densities, the gas temperature
        or the electric field are changed.

        Parameters
        ----------

        Returns
        -------

        Notes
        -----
        The most expensive calculations in this method are cached so they are
        not repeated in each call.  Therefore the execution time may vary
        wildly in different calls.  It takes very long whenever you change
        the solver's grid; therefore is is strongly recommended not to 
        change the grid if is not strictly neccesary.

        """

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
        """ Calculates a Maxwell-Boltzmann distribution function.

        Parameters
        ----------
        kT : float
           The electron temperature in eV.

        Returns
        -------
        f : array of floats 
           A normalized Boltzmann-Maxwell EEDF with the given temperature.

        Notes
        -----
        This is often useful to give a starting value for the EEDF.
        """

        return (2 * np.sqrt(1 / np.pi)
                * kT**(-3./2.) * np.exp(-self.cenergy / kT))


    def iterate(self, f0, delta=1e14):
        """ Iterates once the EEDF. 

        Parameters
        ----------
        f0 : array of floats
           The previous EEDF
        delta : float
           The convergence parameter.  Generally a larger delta leads to faster
           convergence but a too large value may lead to instabilities or
           slower convergence.

        Returns
        -------
        f1 : array of floats
           A new value of the distribution function.

        Notes
        -----
        This is a low-level routine not intended for normal uses.  The
        standard entry point for the iterative solution of the EEDF is
        the :func:`BoltzmannSolver.converge` method.
        """

        A, Q = self._linsystem(f0)

        f1 = spsolve(sparse.eye(self.n) 
                     + delta * A - delta * Q, f0)

        return self._normalized(f1)

    
    def converge(self, f0, maxn=100, rtol=1e-5, delta0=1e14, m=4.0,
                 full=False, **kwargs):
        """ Iterates and attempted EEDF until convergence is reached.

        Parameters
        ----------
        f0 : array of floats
           Initial EEDF.
        maxn : int
           Maximum number of iteration until the convergence is declared as 
           failed (default: 100).
        rtol : float
           Target tolerance for the convergence.  The iteration is stopped
           when the difference between EEDFs is smaller than rtol in L1
           norm (default: 1e-5).
        delta0 : float
           Initial value of the iteration parameter.  This parameter
           is adapted in succesive iterations to improve convergence.
           (default: 1e14)
        m : float
           Attempted reduction in the error for each iteration.  The Richardson
           extrapolation attempts to reduce the error by a factor m in each
           iteration.  Larger m means faster convergence but also possible
           instabilities and non-decreasing errors. (default: 4)
        full : boolean
           If true returns convergence information besides the EEDF.

        Returns
        -------
        f1 : array of floats
           Final EEDF
        iters : int (returned only if ``full`` is True)
           Number of iterations required to reach convergence.
        err : float (returned only if ``full`` is True)
           Final error estimation of the EEDF (must me smaller than ``rtol``).

        Notes
        -----
        If convergence is not achieved after ``maxn`` iterations, an exception
        of type ``ConvergenceError`` is raised.
        """

        err0 = err1 = 0
        delta = delta0

        for i in range(maxn):
            # If we have already two error estimations we use Richardson
            # extrapolation to obtain a new delta and speed up convergence.
            if 0 < err1 < err0:
                # Linear extrapolation
                # delta = delta * err1 / (err0 - err1)

                # Log extrapolation attempting to reduce the error a factor m
                delta = delta * np.log(m) / (np.log(err0) - np.log(err1))
                
            f1 = self.iterate(f0, delta=delta, **kwargs)
            err0 = err1
            err1 = self._norm(abs(f0 - f1))
            
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


    def _linsystem(self, F):
        """ Constructs the matrix and the right hand side of the linear system depending on the growth model
        growth_model == 0 (None), 1 (Temporal), 2 (Spatial) """
        Q = self._PQ(F)

        # Useful for debugging but wasteful in normal times.
        # if np.any(np.isnan(Q.todense())):
        #     raise ValueError("NaN found in Q")

        nu = np.sum(Q.dot(F))

        if (self.growth_model==0):
            A = self._scharf_gummel(self.sigma_m)

        elif (self.growth_model==1):
            sigma_tilde = self.sigma_m + nu / np.sqrt(self.benergy) / GAMMA

            # The R (G) term, which we add to A.
            G = 2 * self.denergy32 * nu / 3

            if (self.coulomb): self._coulomb(F)

            A = self._scharf_gummel(sigma_tilde, G)

        elif (self.growth_model==2):
            dF = np.r_[0, np.diff(F) / np.diff(self.cenergy), 0]

            sigma_m_c = 0.5 * (self.sigma_m[:-1] + self.sigma_m[1:])

            mu = - GAMMA / 3 * ( integrate.simps(self.benergy*dF/self.sigma_m, x=self.benergy) ) 
            D = GAMMA / 3 * ( integrate.simps(self.cenergy*F/sigma_m_c, x=self.cenergy) ) 
            alpha = (mu*self.EN - np.sqrt((mu*self.EN)**2 - 4*D*nu)) / 2 / D

            G = - alpha * GAMMA / 3 * (alpha * (self.benergy[1:]**2 -
                self.benergy[:-1]**2) / sigma_m_c / 2 -
                self.EN * (self.benergy[1:]/self.sigma_m[1:] - self.benergy[:-1]/self.sigma_m[:-1]))

            self.WS = - 2 / 3 * GAMMA * alpha * self.EN * self.benergy / self.sigma_m

            if (self.coulomb): self._coulomb(F)

            A = self._scharf_gummel(self.sigma_m, G)

        # if np.any(np.isnan(A.todense())):
        #     raise ValueError("NaN found in A")

        return A, Q


    def _norm(self, f):
        return integrate.simps(f * np.sqrt(self.cenergy), x=self.cenergy)
        
        # return np.sum(f * np.sqrt(self.cenergy) * self.denergy)

    def _normalized(self, f):
        N = self._norm(f)
        return f / N


    def _scharf_gummel(self, sigma_tilde, G=0):
        D = self.DA / (sigma_tilde) + self.DB
        
        # Due to the zero flux b.c. the values of z[0] and z[-1] are never used.
        # To make sure, we set is a nan so it will taint everything if ever 
        # used.
        # TODO: Perhaps it would be easier simply to set the appropriate
        # values here to satisfy the b.c.
        if self.growth_model<=1:
            if self.coulomb:
                D += self.DC
                z  = (self.W + self.WC) * np.r_[np.nan, np.diff(self.cenergy), np.nan] / D
                a0 = (self.W + self.WC) / (1 - np.exp(-z))
                a1 = (self.W + self.WC) / (1 - np.exp(z))
            else:
                z  = self.W * np.r_[np.nan, np.diff(self.cenergy), np.nan] / D
                a0 = self.W / (1 - np.exp(-z))
                a1 = self.W / (1 - np.exp(z))
        elif self.growth_model==2:
            if self.coulomb:
                D += self.DC
                z  = (self.W + self.WC + self.WS) * np.r_[np.nan, np.diff(self.cenergy), np.nan] / D
                a0 = (self.W + self.WC + self.WS) / (1 - np.exp(-z))
                a1 = (self.W + self.WC + self.WS) / (1 - np.exp(z))
            else:
                z  = (self.W + self.WS) * np.r_[np.nan, np.diff(self.cenergy), np.nan] / D
                a0 = (self.W + self.WS) / (1 - np.exp(-z))
                a1 = (self.W + self.WS) / (1 - np.exp(z))

        diags = np.zeros((3, self.n))

        # No flux at the energy = 0 boundary
        diags[0, 0]  = a0[1]

        diags[0, 1:] =  a0[2:]  - a1[1:-1]
        diags[1, :]  =  a1[:-1]
        diags[2, :]  = -a0[1:]

        # F[n+1] = 2 * F[n] - F[n-1] b.c.
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


    def _g(self, F0):
        Fp = np.r_[F0[0], F0, F0[-1]]
        cenergyp = np.r_[self.cenergy[0], self.cenergy, self.cenergy[-1]]
        g = np.log(Fp[2:] / Fp[:-2]) / (cenergyp[2:] - cenergyp[:-2])
        
        return g


    def _PQ(self, F0, reactions=None):
        PQ = sparse.csr_matrix((self.n, self.n))

        g = self._g(F0)
        if reactions is None:
            reactions = list(self.iter_inelastic())

        data = []
        rows = []
        cols = []
        for t, k in reactions:
            r = t.density * GAMMA * k.scatterings(g, self.cenergy)
            in_factor = k.in_factor
            
            data.extend([in_factor * r, -r])
            rows.extend([k.i, k.j])
            cols.extend([k.j, k.j])

        data, rows, cols = (np.hstack(x) for x in (data, rows, cols))
        PQ = sparse.coo_matrix((data, (rows, cols)),
                              shape=(self.n, self.n))

        return PQ

    def _coulomb(self, F0):
        """ Calculates the coulomb collisions terms as done in the original 2005 paper.
        The calculation is rather costly as it involves three integral calculations """
        kTe = 2. / 3. * co.e * integrate.simps(self.cenergy**1.5 * F0, self.cenergy)
        A1_f = np.sqrt(self.cenergy) * F0
        A1 = np.array([integrate.simps(A1_f[:i+1], 
                        self.cenergy[:i+1]) for i in range(self.n)])
        A2_f = self.cenergy**1.5 * F0
        A2 = np.array([integrate.simps(A2_f[:i+1],
                        self.cenergy[:i+1]) for i in range(self.n)])
        A3 = np.array([integrate.simps(F0[i:],
                        self.cenergy[i:]) for i in range(self.n)])

        coulomb_param = (12. * np.pi * (co.epsilon_0 * kTe)**1.5 /
                            co.e**3 / np.sqrt(self.electron_density))
        a = co.e**2 * GAMMA / 24. / np.pi / co.epsilon_0**2 * np.log(coulomb_param)

        A1 = np.r_[A1[0], 0.5 * (A1[:-1] + A1[1:]), A1[-1]]
        A2 = np.r_[A2[0], 0.5 * (A2[:-1] + A2[1:]), A2[-1]]
        A3 = np.r_[A3[0], 0.5 * (A3[:-1] + A3[1:]), A3[-1]]

        self.WC = - 3. * a * self.ion_degree * A1
        self.DC = 2. * a * self.ion_degree * (A2 + self.benergy**1.5*A3)

    ##
    # Now some functions to calculate rates transport parameters from the
    # converged F0        
    def rate(self, F0, k, weighted=False):
        """ Calculates the rate of a process from a (usually converged) EEDF.

        Parameters
        ----------
        F0 : array of floats
           Distribution function.
        k : :class:`process.Process` or string
           The process whose rate we want to calculate.  If `k` is a string,
           it is passed to :func:`search` to obtain a process instance.
        weighted : boolean, optional
           If true, the rate is multiplied by the density of the target.

        Returns
        -------
        rate : float
           The rate of the given process according to `F0`.

        Examples
        --------
        >>> k_ionization = bsolver.rate(F0, "N2 -> N2^+")


        See Also
        --------
        search : Find a process that matches a given signature.

        """
        g = self._g(F0)

        if isinstance(k, str):
            k = self.search(k)

        k.set_grid_cache(self.grid)

        r = k.scatterings(g, self.cenergy)

        P = sparse.coo_matrix((GAMMA * r, (k.j, np.zeros(r.shape))), 
                              shape=(self.n, 1)).todense()
                              
        P = np.squeeze(np.array(P))

        rate = F0.dot(P)
        if weighted:
            rate *= k.target.density
            
        return rate


    def mobility(self, F0):
        """ Calculates the reduced mobility (mobility * N) from the EEDF.

        Parameters
        ----------
        F0 : array of floats
           The EEDF used to compute the mobility.

        Returns
        -------
        mun : float
           The reduced mobility (mu * n) of the electrons in SI 
           units (V / m / s).

        Examples
        --------
        >>> mun = bsolver.mobility(F0)

        See Also
        --------
        diffusion : Find the reduced diffusion rate from the EEDF.
        """

        DF0 = np.r_[0.0, np.diff(F0) / np.diff(self.cenergy), 0.0]
        Q = self._PQ(F0, reactions=self.iter_growth())

        nu = np.sum(Q.dot(F0)) / GAMMA
        sigma_tilde = self.sigma_m + nu / np.sqrt(self.benergy)

        y = DF0 * self.benergy / sigma_tilde
        y[0] = 0

        return -(GAMMA / 3) * integrate.simps(y, x=self.benergy)


    def diffusion(self, F0):
        """ Calculates the diffusion coefficient from a 
        distribution function.

        Parameters
        ----------
        F0 : array of floats
           The EEDF used to compute the diffusion coefficient.

        Returns
        -------
        diffn : float
           The reduced diffusion coefficient of electrons in SI units..

        See Also
        --------
        mobility : Find the reduced mobility from the EEDF.

        """

        Q = self._PQ(F0, reactions=self.iter_growth())

        nu = np.sum(Q.dot(F0)) / GAMMA

        sigma_m = np.zeros_like(self.cenergy)
        for target, process in self.iter_momentum():
            s = target.density * process.interp(self.cenergy)
            sigma_m += s

        sigma_tilde = sigma_m + nu / np.sqrt(self.cenergy)

        y = F0 * self.cenergy / sigma_tilde

        return (GAMMA / 3) * integrate.simps(y, x=self.cenergy)


    def mean_energy(self, F0):
        """ Calculates the mean energy from a distribution function.

        Parameters
        ----------
        F0 : array of floats
           The EEDF used to compute the diffusion coefficient.

        Returns
        -------
        energy : float
           The mean energy of electrons in the EEDF.

        """

        de52 = np.diff(self.benergy**2.5)
        return np.sum(0.4 * F0 * de52)

