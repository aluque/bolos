import sys
import logging

import numpy as np
import pylab
import scipy.constants as co
import parse
import solver
import grid

from matplotlib import cm

from IPython import embed

def main():
    import json
    import yaml

    logging.basicConfig(format='[%(asctime)s] %(module)s.%(funcName)s: %(message)s', 
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        level=logging.DEBUG)

    with open(sys.argv[1]) as fp:
        processes = parse.parse(fp)

    
    gr = grid.LinearGrid(0, 40., 200)

    bsolver = solver.BoltzmannSolver(gr)
    bsolver.load_collisions(processes)

    bsolver.target['N2'].density = 0.8
    bsolver.target['O2'].density = 0.2

    bsolver.EN = 120 * solver.TOWNSEND
    bsolver.kT = 300 * co.k / co.eV

    bsolver.init()

    f0 = bsolver.maxwell(2.0)
    f1 = bsolver.converge(f0, maxn=50, rtol=1e-7)

    eng, eedf, _ = np.loadtxt("prelim/output.dat",
                              unpack=True, skiprows=2)

    pylab.figure(2)

    pylab.plot(bsolver.cenergy, f1, lw=1.75, c='r')
    pylab.plot(eng, eedf, '--', lw=2.0, c='k')

    pylab.semilogy()
    pylab.xlim([0, 30])
    pylab.ylim([5e-4, 1])
    pylab.show()


if __name__ == '__main__':
    main()
