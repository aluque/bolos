.. _tutorial:

========
Tutorial
========

This tutorial will guide you through all the step that you must follow in order
to use BOLOS in your code to solve the Boltzmann equation.

Installation
^^^^^^^^^^^^

BOLOS is a pure Python package and it sticks to the Python conventions for the
distribution of libraries.  Its only dependencies are NumPy and SciPy.  See `here <http://scipy.org/install.html>`_ for installation instructions of these packages for your operating system.  

There are a few ways to have BOLOS installed in your system:

  1. Download the full source repo from github::

      git clone https://github.com/aluque/bolos.git

     This will create a `bolos` folder with the full code, examples and 
     documentation source.  You can then install bolos by e.g. typing::

       python setup.py install

     Alternatively since BOLOS is pure python package, you can put the `bolos/`
     sub-folder to whatever place where it can be found by the Python 
     interpreter (including your `PYTHONPATH`).

  2. You can use the Python Package Index (PyPI).  From there you can download
     a tarball or you can instruct `pip` to download the package for you
     and install it in your system::

       pip install bolos


First steps
^^^^^^^^^^^

To start using bolos from your Python, import the required modules::

  from bolos import parser, grid, solver

Usually you only need to import these three packages: 

  * `parser` contains methods to parse a file with cross-sections in 
    Bolsig+ format, 
  * `grid` allows you to define different types of grids in energy space.
  * `solver` contains the :class:`solver.BoltzmannSolver`, which the class that
    you will use to solve the Boltzmann equation.

Now you can define an energy grid where you want to evaluate the electron 
energies.  The :module:`grid` contains a few classes to do this.  The simplest
one defines a linear grid.  Let's create a grid extending from 0 to 20 eV with 
200 cells: ::

  gr = grid.LinearGrid(0, 60., 200)

We want to use this grid in a :class:`solver.BoltzmannSolver` instance that
we initialize as::

  boltzmann = solver.BoltzmannSolver(gr)

Loading cross-sections
^^^^^^^^^^^^^^^^^^^^^^

The next step is to load a set of cross-sections for the processes that
will affect the electrons.  BOLOS does not come with any set of 
cross-sections.  You can obtain them from the great database `LxCat <http://fr.lxcat.net/>`_.  BOLOS can read without changes files downloaded from LxCat.

Now let's tell `boltzmann` to load a set of cross-sections from a file named
`lxcat.dat`::

  with open('lxcat.dat') as fp:
        processes = parser.parse(fp)
  boltzmann.load_collisions(processes)

Do not worry if there are processes for species that you do not want to include:
they will be ignored by BOLOS without a performance penalty.

Setting the conditions
^^^^^^^^^^^^^^^^^^^^^^

Now we have to set the conditions in our plasma.  First, we set the molar fractions; for example for synthetic air we do::

  boltzmann.target['N2'].density = 0.8
  boltzmann.target['O2'].density = 0.2

Note that this process requires that you have already loaded cross-sections for
the targets that you are setting.  Also, BOLOS does not check if the
molar fractions add to 1: it is the user's responsibility to select
reasonable molar fractions. 

Next we set the gas temperature and the reduced electric field.  BOLOS
expect a reduced electric field in Vm^2 and a temperature in eV.
However, you can use some predefined constants if you prefer to think
in terms of Kelvin and Townsend.  Here we set a temperature of 300K
and a reduced electric field of 120 Td::

  boltzmann.kT = 300 * solver.KB / solver.ELECTRONVOLT
  boltzmann.EN = 120 * solver.TOWNSEND

After you set these conditions, you must tell BOLOS to update its
internal state to take them into account.  You must do this whenever
you change kT, EN or the underlying grid::

  boltzmann.init()

Obtaining the EEDF
^^^^^^^^^^^^^^^^^^

We have now everything in place to solve the Boltzmann equation.
Since the solver is iterative, we must start with some guess; it does
not make much difference which one as long as it is not too
unreasonable.  For example, we can start with Maxwell-Boltzmann
distribution with a temperature of 2 eV::

  fMaxwell = boltzmann.maxwell(2.0)

Now we ask `boltzmann` to iterate the solution until it is satisfied
that it has converged::

  f = boltzmann.converge(fMaxwell, maxn=100, rtol=1e-5)

Here `maxn` is the maximum number of iterations and `rtol` is the
desired tolerance.  

We now have a distribution function in `f` that is a reasonable
approximation to the exact solution.  However, we made some arbitrary
choices in order to calculate it and perhaps we may still get a more
accurate one.  For example, why did we select a grid from 0 to 60 eV
with 200 cells?  Perhaps we should base our grid on the mean energy of
electrons::

  # Calculate the mean energy according to the first EEDF
  mean_energy = boltzmann.mean_energy(f0)

  # Set a new grid extending up to 15 times the mean energy.
  # Now we use a quadritic grid instead of a linear one.
  newgrid = grid.QuadraticGrid(0, 15 * mean_energy, 200)

  # Set the new grid and update the internal
  boltzmann.grid = newgrid
  boltzmann.init()

  # Calculate an EEDF in the new grid by interpolating the old one
  finterp = boltzmann.grid.interpolate(f, gr)

  # Iterate until we have a new solution
  f1 = boltzmann.converge(finterp, maxn=200, rtol=1e-5)
  

Calculating transport coefficients and reaction rates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Often you are not interested in the EEDF itself but you are working
with a fluid clode and you want to know the transport coefficients and
reaction rates as functions of temperature or E/n.

It's quite easy to obtain the reduced mobility and diffusion rate once
you have the EEDF::

  mun = boltzmann.mobility(f1)
  diffn = boltzmann.diffusion(f1)

This tells you the reduced mobility `mu*n` and diffusion `D*n`, both 
in SI units.

To calculate reaction rates, use :func:`solver.BoltzmannSolver.rate`.  
There are a couple of manners in which you can specify the process.
You can use its signature::

  # Obtain the reaction rate for impact ionization of molecular nitrogen.
  k = boltzmann.rate(f1, "N2 -> N2^+")

This is equivalent to the following sequence::

  proc = boltzmann.search("N2 -> N2^+")[0]
  k = boltzmann.rate(f1, proc)

Here we have first looked in the set of reactions contained in the
`boltzmann` instance for a process matching the signature `"N2 ->
N2^+"`.  :func:`solver.BoltzmannSolver.search` returns a
:class:`process.Process` instance that you can then pass to 
:func:`solver.BoltzmannSolver.rate`.  

The methods :func:`solver.BoltzmannSolver.iter_all`,
:func:`solver.BoltzmannSolver.iter_elastic` and
:func:`solver.BoltzmannSolver.iter_inelastic` let you iterate over the
targets and processes contained in a :class:`solver.BoltzmannSolver`
instance. (These are the processes that we loaded earlier with
:func:`soler.BoltzmannSolver.load_collisions`) ::

  for target, proc in boltzmann.iter_inelastic():
      print "The rate of %s is %g" % (str(proc), boltzmann.rate(f1, proc))



