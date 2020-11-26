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
    E_N = np.linspace(30, 600, 31)
    maxgrid = np.linspace(20, 180, 31)
    inital_energy = np.linspace(5, 12, 31)
    fichier_out = open("IST_Lisbon_air_bolos.dat","w")

    for index, reduced_e in enumerate(E_N):
        fichier_out.write('R%d\n' % (index+1))
        fichier_out.write('------------------------------------------------------------\n')
        fichier_out.write('Electric field / N (Td)                    %.1f \n'% reduced_e )
        fichier_out.write('Gas temperature (K)                        %.1f \n'% temp )
        # Use a linear grid from 0 to 60 eV with 500 intervals.
        gr = grid.LinearGrid(0.001, maxgrid[index], 200)

        # Initiate the solver instance
        bsolver = solver.BoltzmannSolver(gr)

        # Parse the cross-section file in BOSIG+ format and load it into the
        # solver.
        with open(args.input) as fp:
            bsolver.load_collisions(parser.parse(fp))

        # Set the conditions.  And initialize the solver
        bsolver.target['N2'].density = 0.8
        bsolver.target['O2'].density = 0.2
        bsolver.kT = temp * co.k / co.eV
        bsolver.EN = reduced_e * solver.TOWNSEND
        bsolver.init()

        # Start with Maxwell EEDF as initial guess.  Here we are starting with
        # with an electron temperature of 2 eV
        f0 = bsolver.maxwell(2.0)
        # plt.plot(bsolver.grid.c, bsolver.grid.c**(1/2)*f0, label='Init')


        # Solve the Boltzmann equation with a tolerance rtol and maxn iterations.
        f0 = bsolver.converge(f0, maxn=200, rtol=1e-4)
        # plt.plot(bsolver.grid.c, bsolver.grid.c**(1/2)*f0, label='First iteration')

        # Second pass: with an automatic grid and a lower tolerance.
        mean_energy = bsolver.mean_energy(f0)
        newgrid = grid.QuadraticGrid(0, 12 * mean_energy, 200)
        bsolver.grid = newgrid
        bsolver.init()

        f1 = bsolver.grid.interpolate(f0, gr)
        f1 = bsolver.converge(f1, maxn=200, rtol=1e-5)

        # plt.plot(bsolver.grid.c, bsolver.grid.c**(1/2)*f1, label='Second iteration')
        # plt.legend()
        # plt.show()

        # Search for a particular process in the solver and print its rate.
        #k = bsolver.search('N2 -> N2^+')
        #print "THE REACTION RATE OF %s IS %g\n" % (k, bsolver.rate(f1, k))
        
        # You can also iterate over all processes or over processes of a certain
        # type.

        # Calculate the mobility and diffusion.
        fichier_out.write('------------------------------------------------------------\n')
        fichier_out.write("TRANSPORT PARAMETERS:\n")
        fichier_out.write("Mean energy (eV)                          %.4e  \n" % bsolver.mean_energy(f1))
        fichier_out.write("Mobility *N (1/m/V/s)                     %.4e  \n" % bsolver.mobility(f1))
        fichier_out.write("Diffusion coefficient *N (1/m/s)          %.4e  \n" % bsolver.diffusion(f1))
        
        fichier_out.write('------------------------------------------------------------\n')
        fichier_out.write("Rate coefficients (m3/s)\n")
        for t, p in bsolver.iter_all():
            fichier_out.write("%-40s   %.4e\n" % (str(p), bsolver.rate(f1, p)))

        fichier_out.write('------------------------------------------------------------\n')
        fichier_out.write("Energy (eV) EEDF (eV-3/2)\n")
        for i in range(len(bsolver.grid.c)):
            fichier_out.write("%.4e    %.4e\n" % (bsolver.grid.c[i],f1[i]))

        fichier_out.write('\n\n\n')

if __name__ == '__main__':
    main()
