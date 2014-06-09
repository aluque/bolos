.. _faq:


==========================
Frequently Asked Questions
==========================


Why another Boltzmann solver?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The low-temperature plasma community already has 
`BOLSIG+ <http://www.bolsig.laplace.univ-tlse.fr/>`_, a highly optimized, 
user-friendly solver for the Boltzmann equation [HP2005]_.  BOLSIG+ is 
freely distributed
by its authors, Hagelaar and Pitchford.  Why did I start BOLOS, 
another Boltzmann solver based on similar algorithms?

The simplest reply is that, as a BOLSIG+ user, I wanted to understand better 
what goes on beneath BOLSIG+ and the best way to understand something is
to do it yourself.

However, I also felt that an Open Source
implementation would benefit the community.  There are a number of
drawbacks to the way BOLSIG+ is packaged that sometimes limited or
slowed down my own research.  For example, we only have a Windows
version, whereas many scientist now use Linux or Mac OS X as their
platforms of choice.  Also, since BOLSIG+ is distributed only as
binary package, it is difficult or impossible to integrate into other
codes or to make it part of an automated pipeline. 

Finally, there is the old *hacker ethic*, where we tinker with each
other's code and tools and collaborate to improve them.  This is
particularly relevant for scientists, since we all build on the work of
others.  Having an open source, modern, Boltzmann solver may
facilitate new improvements and its integration with other tools.


Why did you use Python?
^^^^^^^^^^^^^^^^^^^^^^^

Because my main purpose was to develop a simple, readable code in the
hope that other people would take it and perhaps improve it.

The code relies on the `Numpy <http://www.numpy.org/>`_ and 
`SciPy <http://www.scipy.org/>`_ libraries that interface with
highly optimized, C or FORTRAN code.  


What version(s) of Python does BOLOS support?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Presently, only 2.7.  In future release, Python 3+ will be supported.
Since BOLOS is a pure Python package, the transition should be 
straightforward.



Can BOLOS read cross-sections in BOLSIG+ format?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes!  You can use your cross-sections files from 
`BOLSIG+ <http://www.bolsig.laplace.univ-tlse.fr/>`_ or from 
`LxCat <http://fr.lxcat.net/>`_
without changes.  Any problem reading these files will be treated as a
bug.



How can I start using it?
^^^^^^^^^^^^^^^^^^^^^^^^^

This is the code required to load a cross-section database from a file 
passed as the first command-line parameter and
calculate reaction rates and the mobility and diffusion coefficients::

  import sys
  import logging
  
  import numpy as np
  import pylab
  import scipy.constants as co
  from bolos import parser, solver, grid
  
  def main():
  
      # Use a linear grid from 0 to 40 eV with 100 intervals.
      gr = grid.LinearGrid(0, 40., 100)
  
      # Initiate the solver instance
      bsolver = solver.BoltzmannSolver(gr)
  
      # Parse the cross-section file in BOSIG+ format and load it into the
      # solver.
      with open(sys.argv[1]) as fp:
          processes = parse.parse(fp)
      bsolver.load_collisions(processes)
  
      # Set the conditions.  And initialize the solver
      bsolver.target['N2'].density = 0.8
      bsolver.target['O2'].density = 0.2
      bsolver.kT = 300 * co.k / co.eV
      bsolver.EN = 100 * solver.TOWNSEND
      bsolver.init()
  
      # Start with Maxwell EEDF as initial guess.  Here we are starting with
      # with an electron temperature of 2 eV
      f0 = bsolver.maxwell(2.0)
  
      # Solve the Boltzmann equation with a tolerance 1e-5 and 50
      max. iteration  s.
      f1 = bsolver.converge(f0, maxn=50, rtol=1e-5)
  
      # Search for a particular process in the solver and print its rate.
      k = bsolver.search('N2 -> N2^+')
      print "THE REACTION RATE OF %s IS %g\n" % (k, bsolver.rate(f1, k))
      
      # You can also iterate over all processes or over processes of a certain
      # type.
      print "\nREACTION RATES OF INELASTIC PROCESSES:\n"
      for t, k in bsolver.iter_inelastic():
          print k, bsolver.rate(f1, k)
  
      # Calculate the mobility and diffusion.
      print "\nTRANSPORT PARAMETERS:\n"
      print "mobility * N  = %g  (1/m/V/s)" % bsolver.mobility(f1)
      print "diffusion * N = %g  (1/m/s)" % bsolver.diffusion(f1)
  
  
  
  
  if __name__ == '__main__':
      main()


This code is distributed in the ``sample.py`` file of the
distribution.  You can invoke it as::

  python sample.py LXCat-June2013.txt



If I use BOLOS for my research, which paper should I cite?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BOLOS follows the algorithm described by Hagelaar and Pitchford so you
should definitely cite their paper [HP2005]_.

There is not yet any publication associated directly with BOLOS, so if
you use it please link to its source repository at github.

.. _BOLSIG+: http://www.bolsig.laplace.univ-tlse.fr/

.. [HP2005] *Solving the Boltzmann equation to obtain electron transport
coefficients and rate coefficients for fluid models*, G. J. M. Hagelaar 
and L. C. Pitchford, Plasma Sources Sci. Technol. **14** (2005)
722–733.



