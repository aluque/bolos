"""
This code compares the reaction rates calculated by BOLOS with those obtained
from Bolsig+.  It produces a series of PDF files with plots comparing these 
rates.  To invoke it, use

  python bolsig.py cs.dat bolsig.dat

where 'cs.dat' is a Bolsig+-compatible file of cross sections and bolsig.dat
is a file containing reaction rates as function of the reduced electric field.
We assume that the first column contains electric field whereas each of the 
following columns contain reaction rates, ordered as in the cs.dat file.

As it is currently written, we calculate rates for synthetic air 
(N2:O2 = 80:20) but you can change that below.
"""

import sys
import logging
import argparse
import os
import itertools

import numpy as np
import scipy.constants as co
from bolos import parser, solver, grid

from matplotlib import pyplot as plt


def main():
    argparser = argparse.ArgumentParser()

    argparser.add_argument("input", 
                        help="File with cross-sections in BOLSIG+ format")
    argparser.add_argument("bolsigdata", 
                        help="File with BOLSIG+ rates")
    argparser.add_argument("--debug", 
                        help="If set, produce a lot of output for debugging", 
                        action='store_true', default=False)
    args = argparser.parse_args()

    if args.debug:
        logging.basicConfig(format='[%(asctime)s] %(module)s.%(funcName)s: '
                            '%(message)s', 
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            level=logging.DEBUG)

    
    # Use a linear grid from 0 to 60 eV with 100 intervals.
    gr = grid.LinearGrid(0, 80., 400)

    # Initiate the solver instance
    bsolver = solver.BoltzmannSolver(gr)

    # Parse the cross-section file in BOSIG+ format and load it into the
    # solver.
    with open(args.input) as fp:
        processes = parser.parse(fp)
    processes = bsolver.load_collisions(processes)
    
    # Set the conditions.  And initialize the solver
    bsolver.target['N2'].density = 0.8
    bsolver.target['O2'].density = 0.2

    # Calculate rates only for N2 and O2
    processes = [p for p in processes if p.target.density > 0]

    bolsig = np.loadtxt(args.bolsigdata)

    # Electric fields in the BOLSIG+ file
    en = bolsig[:, 1]

    # Create an array to store our results
    bolos = np.empty((len(en), len(processes)))

    # Start with Maxwell EEDF as initial guess.  Here we are starting with
    # with an electron temperature of 2 eV
    f0 = bsolver.maxwell(2.0)


    # Solve for each E/n
    for i, ien in enumerate(en):
        bsolver.grid = gr

        logging.info("E/n = %f Td" % ien)
        bsolver.EN = ien * solver.TOWNSEND
        bsolver.kT = 300 * co.k / co.eV
        bsolver.init()

        # Note that for each E/n we start with the previous E/n as
        # a reasonable first guess.
        f1 = bsolver.converge(f0, maxn=200, rtol=1e-4)

        # Second pass: with an automatic grid and a lower tolerance.
        mean_energy = bsolver.mean_energy(f1)
        newgrid = grid.QuadraticGrid(0, 15 * mean_energy, 200)
        bsolver.grid = newgrid
        bsolver.init()

        f2 = bsolver.grid.interpolate(f1, gr)
        f2 = bsolver.converge(f2, maxn=200, rtol=1e-5)

        bolos[i, :] = [bsolver.rate(f2, p) for p in processes]
    
    for k, p in enumerate(processes):
        print "%.3d:  %-40s   %s eV" % (k, p, p.threshold or 0.0)
        
        plt.clf()
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)

        plt.plot(en, bolos[:, k], lw=2.0, c='r', label="BOLOS")
        plt.plot(en, bolsig[:, k + 3], lw=2.0, ls='--', c='k', 
                   label="Bolsig+")
        plt.ylabel("k (cm$^\mathdefault{3}$/s)")
        plt.grid()
        plt.semilogy()
        plt.legend(loc='lower right')
        plt.gca().xaxis.set_ticklabels([])

        plt.subplot(2, 1, 2)
        plt.plot(en, abs(bolos[:, k] - bolsig[:, k + 3]) / bolsig[:, k + 3], 
                   lw=2.0, c='b')

        plt.xlabel("E/n (Td)")
        plt.ylabel("$|k_{BOLOS}-k_{Bolsig}|/k_{Bolsig}$")
        plt.grid()
        plt.semilogy()
        plt.text(0.025, 0.05, 
                   str(p) + " %s eV" % p.threshold,
                   color='b',
                   horizontalalignment='left',
                   verticalalignment='bottom',
                   size=18,
                   transform=plt.gca().transAxes)

        plt.savefig("%.3d.pdf" % k)
        plt.close()



if __name__ == '__main__':
    main()
