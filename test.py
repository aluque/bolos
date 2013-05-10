import sys
import logging

import pylab
import scipy.constants as co
import parse
import solver

def main():
    import json
    import yaml

    logging.basicConfig(format='[%(asctime)s] %(module)s.%(funcName)s: %(message)s', 
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        level=logging.DEBUG)

    with open(sys.argv[1]) as fp:
        processes = parse.parse(fp)

    bsolver = solver.BoltzmannSolver(80.0, n=128)
    bsolver.load_collisions(processes)

    bsolver.target['N2'].density = 0.8
    bsolver.target['O2'].density = 0.2

    bsolver.EN = 120 * solver.TOWNSEND
    bsolver.kT = 300 * co.k / co.eV

    bsolver.init()
    pexc = bsolver.target['N2'].combined_process('EXCITATION')

    for p in bsolver.total.inelastic:
        print str(p), p.threshold

    f0 = bsolver.maxwell(10.0)
    f1, minval, d = bsolver.iterate(f0)

    pylab.plot(bsolver.cenergy, f0, lw=2.0, c='k')
    pylab.plot(bsolver.cenergy, f1, lw=2.0, c='r')

    pylab.show()

    

if __name__ == '__main__':
    main()
