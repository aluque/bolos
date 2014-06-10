.. _faq:


==========================
Frequently Asked Questions
==========================


Why another Boltzmann solver?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The low-temperature plasma community already has 
`BOLSIG+ <http://www.bolsig.laplace.univ-tlse.fr/>`_, a highly optimized, 
user-friendly solver for the Boltzmann equation [HP2005]_.  BOLSIG+ is 
freely distributed by its authors, Hagelaar and Pitchford.  Why did I write 
BOLOS, another Boltzmann solver based on similar algorithms?

The simplest reply is that, as a BOLSIG+ user, I wanted to understand better 
what goes on beneath BOLSIG+ and the best way to understand something is
to do it yourself.

However, I also felt that an Open Source
implementation would benefit the community.  There are a number of
drawbacks to the way BOLSIG+ is packaged that sometimes limited or
slowed down my own research.  For example, we only have a Windows
version, whereas many of us now use Linux or Mac OS X as their
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


How fast is BOLOS?
^^^^^^^^^^^^^^^^^^

I would say it's ``reasobaly fast``.  It takes a few tenths of a second to 
solve the Boltzmann equation.  The code was heavily optimized to use numpy's
and scipy's features, particularly regarding sparse matrices.


Are results the same as with BOLSIG+?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In most cases the difference in reaction rates or transport parameters is 
between 0.1% and 1%.  My guess is that most of the difference comes from the
use of different grids but probably the growth-renormalization term is 
implemented differently (Hagelaar and Pitchford are not very clear on this 
point).

The code in `samples/bolsig.py` helps to compare the two codes.


Feature X is not implemented: what can I do?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes, there are still many features that are not implemented in BOLOS.  
In particular, only the temporal growth model is implemented and many parameters obtained from the EEDF are not yet implemented.  I hope
to add these things gradually.  If you are interested in a particular 
feature you can give it a shot:  pull requests are welcome.  Or you can write 
me and I promise that I will look into it... but you know how tight all our agendas are.


If I use BOLOS for my research, which paper should I cite?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BOLOS follows the algorithm described by Hagelaar and Pitchford so you
should definitely cite their paper [HP2005]_.

There is not yet any publication associated directly with BOLOS, so if
you use it please link to its `source code`_ at github.

.. _BOLSIG+: http://www.bolsig.laplace.univ-tlse.fr/
.. _source code: https://github.com/aluque/bolos
.. [HP2005] *Solving the Boltzmann equation to obtain electron transport
coefficients and rate coefficients for fluid models*, G. J. M. Hagelaar 
and L. C. Pitchford, Plasma Sources Sci. Technol. **14** (2005)
722–733.



