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
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as co
sys.path.append('/Users/cheng/CODE/BOLTZMANN_SOLVER/bolos-master')
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

    print(args.temp)

    if args.debug:
        logging.basicConfig(format='[%(asctime)s] %(module)s.%(funcName)s: '
                            '%(message)s', 
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            level=logging.DEBUG)

    temp = 300
    list_ion_degree = [0, 1e-6, 1e-5, 1e-4, 1e-3]
    # list_ion_degree = [1e-4, 1e-3]
    list_EN = np.geomspace(0.1, 600, num=20)

    # Initialize the solver
    gr = grid.LinearGrid(0, 60, 200)
    bsolver = solver.BoltzmannSolver(gr)
    # Parse the cross-section file in BOSIG+ format and load it into the
    # solver.
    with open(args.input) as fp:
        bsolver.load_collisions(parser.parse(fp))
    bsolver.target['Ar'].density = 1.0
    bsolver.kT = temp * co.k / co.eV


    for ion_degree in list_ion_degree:
        mean_epsilon = []
        ion_rate = []
        for index, EN in enumerate(list_EN):
            print("Ion degree = %.1e\nE/N = %.2f" % (ion_degree, EN))

            # Set the conditions.  And initialize the solver
            # if index != 0:
            #     gr = newgrid
            # else:
            #     gr = grid.LinearGrid(0, 60, 200)

            gr = grid.LinearGrid(0, 60, 200)

            bsolver.grid = gr
            bsolver.EN = EN * solver.TOWNSEND
            bsolver.coulomb = True
            bsolver.electron_density = 1e18
            bsolver.ion_degree = ion_degree
            bsolver.growth_model = 1
            bsolver.init()

            # if index == 0:
            #     f2 = bsolver.maxwell(2.0)
            if index == 0:
                f1 = bsolver.maxwell(1.0)

            # Solve the Boltzmann equation with a tolerance rtol and maxn iterations.
            f1 = bsolver.converge(f1, maxn=200, rtol=1e-6, m=8.0, delta0=2e14)
            # Second pass: with an automatic grid and a lower tolerance.
            # mean_energy = bsolver.mean_energy(f1)
            # print("Mean energy first pass = %.2f" % mean_energy)
            # # newgrid = grid.QuadraticGrid(0, 15 * mean_energy, 400)
            # newgrid = grid.LinearGrid(0, 60, 200)
            # bsolver.grid = newgrid
            # bsolver.init()

            # f2 = bsolver.grid.interpolate(f1, gr)
            # # f2 = bsolver.converge(f2, maxn=200, rtol=1e-6, m=8.0, delta0=2e14)
            # f2 = bsolver.converge(f2, maxn=200, rtol=1e-6, m=4.0, delta0=1e14)

            # mean_epsilon.append(bsolver.mean_energy(f2))
            # for t, p in bsolver.iter_inelastic():
            #     if p.kind[:3] == 'ION':
            #         ion_rate.append(bsolver.rate(f2, p))

            mean_epsilon.append(bsolver.mean_energy(f1))
            for t, p in bsolver.iter_inelastic():
                if p.kind[:3] == 'ION':
                    ion_rate.append(bsolver.rate(f1, p))
            print("Mean energy = %.2f\nIonization rate = %.2e" % (mean_epsilon[-1], ion_rate[-1]))

        plt.plot(mean_epsilon, ion_rate, label='n/N = %.1e' % ion_degree)

        # Search for a particular process in the solver and print its rate.
        #k = bsolver.search('N2 -> N2^+')
        #print "THE REACTION RATE OF %s IS %g\n" % (k, bsolver.rate(f1, k))
        
        # You can also iterate over all processes or over processes of a certain
        # type.

    plt.legend()
    plt.xlim([1, 10])
    plt.ylim([1e-21, 1e-14])
    plt.yscale('log')
    plt.xlabel(r'$\epsilon$ (eV)')
    plt.ylabel(r'Ionization rate coefficient (m$^3$/s)')
    plt.savefig('FIGURES/Ar_coulomb_rates', bbox_inches='tight')

    plt.savefig('FIGURES/Ar_10Td_coulomb_2it.png', bbox_inches='tight')
if __name__ == '__main__':
    main()
