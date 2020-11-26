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
from scipy import integrate

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

    if args.debug:
        logging.basicConfig(format='[%(asctime)s] %(module)s.%(funcName)s: '
                            '%(message)s', 
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            level=logging.DEBUG)

    temp = 300
    growth_model_string = ['None', 'Temporal', 'Spatial']
    list_growth_model = [0, 1, 2]
    list_EN = np.linspace(30, 600, 30)
    color = ['k', 'b', 'g']

    #search for the mean of the elastic cross sections in log energy space 
    # gr = grid.LinearGrid(0, 120, 200)
    # bsolver = solver.BoltzmannSolver(gr)
    # with open(args.input) as fp:
    #     bsolver.load_collisions(parser.parse(fp))
    # bsolver.target['Ar'].density = 1.0
    # list_kT = np.linspace(1, 15, 15)
    # list_mean_cs = []
    # for t, p in bsolver.iter_all():
    #     if p.kind[:3] == 'ELA':
    #         mean_en_cs = np.max(p.y)
    #         for kT in list_kT:
    #             maxwell =  (2 * np.sqrt(1 / np.pi) * kT**(-3./2.) * np.exp(-p.x / kT))
    #             mean_en_cs = integrate.simps(maxwell * p.y * np.sqrt(p.x), x=p.x)
    #             list_mean_cs.append(mean_en_cs)

    # mean_en_cs = np.max(np.array(list_mean_cs))
    # print(mean_en_cs)
    #             # print('kT = %.1f, mean_cs = %.2e' % (kT, mean_en_cs))
    # print(temp * co.k / co.e * 1.5)
    # plt.plot(list_EN, 2 / 3 * list_EN * np.sqrt(np.pi / 12 / 1e-3) / mean_en_cs * 1e-21 + temp * co.k / co.e * 1.5)
    # plt.show()
    # list_initial_energy = 2 / 3 * list_EN * np.sqrt(np.pi / 12 / 1e-3) / mean_en_cs * 1e-21 + temp * co.k / co.e * 1.5
        # print('EN = %.2f, epsilon = %.2f' % (EN, mean_energy))

    # intial energy at 5eV for all
    list_initial_energy = 5 * np.ones(len(list_EN))

    for growth_model in list_growth_model:
        mean_epsilon = []
        ion_rate = []
        for i, EN in enumerate(list_EN):
            print("Growth model = %s \nE/N = %.2f Td\nInitial energy = %.2f" % (growth_model_string[growth_model], EN, list_initial_energy[i]))
            # fichier_out.write('R%d\n' % (index+1))
            # fichier_out.write('------------------------------------------------------------\n')
            # fichier_out.write('Electric field / N (Td)                    %.1f \n'% reduced_e )
            # fichier_out.write('Gas temperature (K)                        %.1f \n'% temp )
            # Use a linear grid from 0 to 60 eV with 500 intervals.
            # gr = grid.LinearGrid(0, 120, 200)
            # gr = grid.QuadraticGrid(0, maxgrid[index], 400)
            mean_energy = list_initial_energy[i]
            gr = grid.LinearGrid(0, 12 * mean_energy, 200)
            # Initiate the solver instance
            bsolver = solver.BoltzmannSolver(gr)

            # Parse the cross-section file in BOSIG+ format and load it into the
            # solver.
            with open(args.input) as fp:
                bsolver.load_collisions(parser.parse(fp))

            # Set the conditions.  And initialize the solver
            bsolver.target['Ar'].density = 1.0
            bsolver.kT = temp * co.k / co.eV
            bsolver.EN = EN * solver.TOWNSEND
            bsolver.coulomb = False
            bsolver.growth_model = growth_model
            bsolver.init()

            # plot_cross_sections(bsolver, 'cross_sections_Ar_Phelps')

            # Start with Maxwell EEDF as initial guess.  Here we are starting with
            # with an electron temperature of 2 eV
            f0 = bsolver.maxwell(mean_energy)

            # ax.plot(bsolver.grid.c, f0, label='Init')
            # Solve the Boltzmann equation with a tolerance rtol and maxn iterations.
            f0 = bsolver.converge(f0, maxn=200, rtol=1e-4)
            # plt.plot(bsolver.grid.c, f0, label='f0 growth_model = %d' % growth_model)
            # ax.plot(bsolver.grid.c, f0, label='First convergence')
            # Second pass: with an automatic grid and a lower tolerance.
            mean_energy = bsolver.mean_energy(f0)
            print('Mean Energy = %.2e' % mean_energy)
            # newgrid = grid.QuadraticGrid(0, 10 * mean_energy, 400)
            newgrid = grid.QuadraticGrid(0, 12 * mean_energy, 800)
            bsolver.grid = newgrid
            bsolver.init()

            f1 = bsolver.grid.interpolate(f0, gr)
            f1 = bsolver.converge(f1, maxn=200, rtol=1e-6)
            # plt.plot(bsolver.grid.c, f1, color=color[growth_model], label='Growth model = %s' % growth_model_string[growth_model])

            mean_energy = bsolver.mean_energy(f1)
            mean_epsilon.append(mean_energy)

            for t, p in bsolver.iter_inelastic():
                if p.kind[:3]=='ION':
                    ion_rate.append(bsolver.rate(f1, p))

            #update for the spatial pass
            list_initial_energy[i] = 0.4 * mean_energy

        # print(mean_epsilon)
        # print(ion_rate)
        plt.plot(mean_epsilon, ion_rate, color=color[growth_model], label='Growth model = %s' % growth_model_string[growth_model])
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
    plt.legend()
    plt.xlim([5, 10])
    plt.ylim([1e-18, 1e-14])
    plt.yscale('log')
    plt.xlabel(r'$\epsilon$ (eV)')
    plt.ylabel(r'Ionization rate coefficient (m$^3$/s)')
    plt.savefig('FIGURES/Ar_growth_rates', bbox_inches='tight')

if __name__ == '__main__':
    main()
