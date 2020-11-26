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

def plot_cross_sections(solver, figname, x1=1e-3, x2=1e4, y1=1e-23, y2=1e-18):
    plt.figure(figsize=(10, 10))
    for target, process in solver.iter_all():
        plt.plot(process.x, process.y, label='%s %s' % (process.target_name, process.kind))

    plt.xlim([x1, x2])
    plt.ylim([y1, y2])
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'$\epsilon$ [eV]')
    plt.ylabel(r'$\sigma$ [m$^2$]')
    plt.savefig('FIGURES/%s' % figname, bbox_inches='tight')

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

    # fig, ax = plt.subplots()

    args = parse_args()

    print(args.temp)

    if args.debug:
        logging.basicConfig(format='[%(asctime)s] %(module)s.%(funcName)s: '
                            '%(message)s', 
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            level=logging.DEBUG)

    temp = 300
    # E_N = [0.1, 20, 100]
    E_N = 600
    growth_model_string = ['None', 'Temporal', 'Spatial']
    list_growth_model = [0, 1, 2]
    color = ['k', 'b', 'g']
    # list_growth_model = [1]
    # fichier_out = open("Ar_10_EEDF_coulomb.dat","w")

    for growth_model in list_growth_model:
        print("growth_model = %d" % growth_model)
        # fichier_out.write('R%d\n' % (index+1))
        # fichier_out.write('------------------------------------------------------------\n')
        # fichier_out.write('Electric field / N (Td)                    %.1f \n'% reduced_e )
        # fichier_out.write('Gas temperature (K)                        %.1f \n'% temp )
        # Use a linear grid from 0 to 60 eV with 500 intervals.
        gr = grid.LinearGrid(0, 120, 200)
        # gr = grid.QuadraticGrid(0, maxgrid[index], 400)

        # Initiate the solver instance
        bsolver = solver.BoltzmannSolver(gr)

        # Parse the cross-section file in BOSIG+ format and load it into the
        # solver.
        with open(args.input) as fp:
            bsolver.load_collisions(parser.parse(fp))

        # Set the conditions.  And initialize the solver
        bsolver.target['Ar'].density = 1.0
        bsolver.kT = temp * co.k / co.eV
        bsolver.EN = E_N * solver.TOWNSEND
        bsolver.coulomb = False
        bsolver.growth_model = growth_model
        bsolver.init()

        # plot_cross_sections(bsolver, 'cross_sections_Ar_Phelps')

        # Start with Maxwell EEDF as initial guess.  Here we are starting with
        # with an electron temperature of 2 eV
        f0 = bsolver.maxwell(8)

        # ax.plot(bsolver.grid.c, f0, label='Init')
        # Solve the Boltzmann equation with a tolerance rtol and maxn iterations.
        f0 = bsolver.converge(f0, maxn=200, rtol=1e-5)
        # plt.plot(bsolver.grid.c, f0, label='f0 growth_model = %d' % growth_model)
        # ax.plot(bsolver.grid.c, f0, label='First convergence')
        # Second pass: with an automatic grid and a lower tolerance.
        mean_energy = bsolver.mean_energy(f0)
        print('Mean Energy = %.2e' % mean_energy)
        # newgrid = grid.QuadraticGrid(0, 10 * mean_energy, 400)
        newgrid = grid.QuadraticGrid(0, 12 * mean_energy, 1000)
        bsolver.grid = newgrid
        bsolver.init()

        f1 = bsolver.grid.interpolate(f0, gr)
        f1 = bsolver.converge(f1, maxn=100, rtol=1e-6)
        plt.plot(bsolver.grid.c, f1, color=color[growth_model], label='Growth model = %s' % growth_model_string[growth_model])

        # Search for a particular process in the solver and print its rate.
        #k = bsolver.search('N2 -> N2^+')
        #print "THE REACTION RATE OF %s IS %g\n" % (k, bsolver.rate(f1, k))
        
        # You can also iterate over all processes or over processes of a certain
        # type.

        # # Calculate the mobility and diffusion.
        # fichier_out.write('------------------------------------------------------------\n')
        # fichier_out.write("TRANSPORT PARAMETERS:\n")
        # fichier_out.write("Mean energy (eV)                          %.4e  \n" % bsolver.mean_energy(f1))
        # fichier_out.write("Mobility *N (1/m/V/s)                     %.4e  \n" % bsolver.mobility(f1))
        # fichier_out.write("Diffusion coefficient *N (1/m/s)          %.4e  \n" % bsolver.diffusion(f1))
        
        # fichier_out.write('------------------------------------------------------------\n')
        # fichier_out.write("Rate coefficients (m3/s)\n")
        # for t, p in bsolver.iter_all():
        #     fichier_out.write("%-40s   %.4e\n" % (str(p), bsolver.rate(f1, p)))

        # fichier_out.write('------------------------------------------------------------\n')
        # fichier_out.write("Energy (eV) EEDF (eV-3/2)\n")
        # for i in range(len(bsolver.grid.c)):
        #     fichier_out.write("%.4e    %.4e\n" % (bsolver.grid.c[i],f1[i]))

        # fichier_out.write('\n\n\n')

        # ax.plot(bsolver.grid.c, f1, label='Final Solution')

    # ax.set_yscale('log')
    # ax.set_xlabel(r'$\epsilon$ (eV)')
    # ax.set_ylabel(r'$F_0$ (eV$^{-3/2}$)')
    # ax.set_xlim([0, 35])
    # ax.set_ylim([1e-9, 1])
    # ax.legend()
    plt.xlim([0, 120])
    plt.yscale('log')
    plt.ylim([1e-9, 1e-1])
    plt.legend()
    plt.xlabel(r'$\epsilon$ (eV)')
    plt.ylabel(r'$F_0$ (eV$^{-3/2}$)')
    plt.savefig('FIGURES/Ar_600Td_growth.png', bbox_inches='tight')

if __name__ == '__main__':
    main()
