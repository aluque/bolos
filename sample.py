import sys
import logging
import argparse

import numpy as np
import scipy.constants as co
from bolos import parse, solver, grid

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("input", 
                        help="File with cross-sections in BOLSIG+ format")
    parser.add_argument("--debug", 
                        help="If set, produce a lot of output for debugging", 
                        action='store_true', default=False)
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(format='[%(asctime)s] %(module)s.%(funcName)s: '
                            '%(message)s', 
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            level=logging.DEBUG)


    # Use a linear grid from 0 to 60 eV with 100 intervals.
    gr = grid.LinearGrid(0, 60., 200)

    # Initiate the solver instance
    bsolver = solver.BoltzmannSolver(gr)

    # Parse the cross-section file in BOSIG+ format and load it into the
    # solver.
    with open(args.input) as fp:
        processes = parse.parse(fp)
    bsolver.load_collisions(processes)

    # Set the conditions.  And initialize the solver
    bsolver.target['N2'].density = 0.8
    bsolver.target['O2'].density = 0.2
    bsolver.kT = 300 * co.k / co.eV
    bsolver.EN = 300.0 * solver.TOWNSEND
    bsolver.init()

    # Start with Maxwell EEDF as initial guess.  Here we are starting with
    # with an electron temperature of 2 eV
    f0 = bsolver.maxwell(2.0)

    # Solve the Boltzmann equation with a tolerance rtol and maxn iterations.
    f1 = bsolver.converge(f0, maxn=50, rtol=1e-5)

    # Search for a particular process in the solver and print its rate.
    #k = bsolver.search('N2 -> N2^+')
    #print "THE REACTION RATE OF %s IS %g\n" % (k, bsolver.rate(f1, k))
    
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
