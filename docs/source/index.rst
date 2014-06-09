.. bolos documentation master file, created by
   sphinx-quickstart on Sun Jun  1 23:36:43 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _root:

=======================
Documentation for bolos
=======================

``BOLOS`` is a BOLtzmann equation solver Open Source library.  

This package provides a pure Python library for the solution of the 
Boltzmann equation for electrons in a non-thermal plasma.  It builds upon
previous work, mostly by G. J. M. Hagelaar and L. C. Pitchford [HP2005]_, 
who developed `BOLSIG+`_.  ``BOLOS`` is a multiplatform, open source 
implementation of a similar algorithm compatible with the `BOLSIG+`_ 
cross-section input format.


The code was developed by `Alejandro Luque <http://www.iaa.es/~aluque>`_ at the 
`Instituto de Astrofísica de Andalucía <http://www.iaa.es>`_ (IAA), `CSIC <http://www.csic.es>`_ and is released under the `LGPLv2 <>`_ license.  

The code is licensed under the `LGPLv2 License`_. Packages can be 
downloaded from the project `homepage`_ on PyPI. The 
`source code`_ can be obtained from
GitHub, which also hosts the `bug tracker`_. The `documentation`_  can be
read on ReadTheDocs.

Contents:

.. toctree::
   :maxdepth: 2
   :numbered:

   install
   tutorial
   recipes
   faq
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _LGPLv2 License: http://www.gnu.org/licenses/lgpl-2.0.html
.. _BOLSIG+: http://www.bolsig.laplace.univ-tlse.fr/
.. _homepage: http://pypi.python.org/pypi/picamera/
.. _documentation: http://picamera.readthedocs.org/
.. _source code: https://github.com/waveform80/picamera
.. _bug tracker: https://github.com/waveform80/picamera/issues
.. [HP2005] *Solving the Boltzmann equation to obtain electron transport coefficients and rate coefficients for fluid models*, G. J. M. Hagelaar and L. C. Pitchford, Plasma Sources Sci. Technol. **14** (2005) 722–733.
