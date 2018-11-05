""" This is an example for the use of the BOLOS Boltzmann solver library.

Use it to obtain reaction rates and transport parameters for a given
reduced electric field and gas temperature.  For example, to read
cross sections from a file named 'cs.dat' and solve
the Boltzmann equation for E/n = 120 Td and a gas temperature of
300 K, call it as

   python single.py cs.dat --en=120 --temp=300


(c) Alejandro Luque Estepa, 2014

"""

import sys
import logging
import argparse

import numpy as np
import scipy.constants as co
from bolos import parser, solver, grid

def parse_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument("input", 
                        help="File with cross-sections in BOLSIG+ format")
    argparser.add_argument("--debug", 
                        help="If set, produce a lot of output for debugging", 
                        action='store_true', default=False)
    argparser.add_argument("--en", 
                           help="Reduced field (in Td)", 
                           type=float, default=100.0)
    argparser.add_argument("--temp", "-T", 
                           help="Gas temperature (in K)", 
                           type=float, default=300.0)

    args = argparser.parse_args()
    return args

def main():
    args = parse_args()

    if args.debug:
        logging.basicConfig(format='[%(asctime)s] %(module)s.%(funcName)s: '
                            '%(message)s', 
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            level=logging.DEBUG)


    # Use a linear grid from 0 to 60 eV with 500 intervals.
    gr = grid.LinearGrid(0, 60., 500)

    # Initiate the solver instance
    bsolver = solver.BoltzmannSolver(gr)

    # Parse the cross-section file in BOSIG+ format and load it into the
    # solver.
    with open(args.input) as fp:
        bsolver.load_collisions(parser.parse(fp))

    # Set the conditions.  And initialize the solver
    bsolver.target['N2'].density = 0.8
    bsolver.target['O2'].density = 0.2
    bsolver.kT = args.temp * co.k / co.eV
    bsolver.EN = args.en * solver.TOWNSEND
    bsolver.init()

    # Start with Maxwell EEDF as initial guess.  Here we are starting with
    # with an electron temperature of 2 eV
    f0 = bsolver.maxwell(2.0)

    # Solve the Boltzmann equation with a tolerance rtol and maxn iterations.
    f0 = bsolver.converge(f0, maxn=200, rtol=1e-4)

    # Second pass: with an automatic grid and a lower tolerance.
    mean_energy = bsolver.mean_energy(f0)
    newgrid = grid.QuadraticGrid(0, 150 * mean_energy, 200)
    bsolver.grid = newgrid
    bsolver.init()

    f1 = bsolver.grid.interpolate(f0, gr)
    f1 = bsolver.converge(f1, maxn=200, rtol=1e-5)

    # Search for a particular process in the solver and print its rate.
    k = bsolver.search('N2 -> N2^+')
    print "THE REACTION RATE OF %s IS %g\n" % (k, bsolver.rate(f1, k))
    print "THE INVERSE REACTION RATE OF %s IS %g\n" % (k, bsolver.inverse_rate(f1, k))
    
    # You can also iterate over all processes or over processes of a certain
    # type.
    print "\nREACTION RATES:\n"
    for t, p in bsolver.iter_all():
        print "%-40s   %.3e m^3/s" % (str(p), bsolver.rate(f1, p))

    # Calculate the mobility and diffusion.
    print "\nTRANSPORT PARAMETERS:\n"
    print "mobility * N   = %.3e  1/m/V/s" % bsolver.mobility(f1)
    print "diffusion * N  = %.3e  1/m/s" % bsolver.diffusion(f1)
    print "average energy = %.3e  eV" % bsolver.mean_energy(f1)
    print "inelastic heating / N = %.3e  eV m^3/s" % bsolver.inelastic_power_loss(f1)
    print "ohmic heating / N = %.3e  eV m^3/s" % (bsolver.mobility(f1) * bsolver.EN * bsolver.EN)

    import pylab
    pylab.plot(bsolver.grid.c, f1)
    pylab.show()
    

if __name__ == '__main__':
    main()
