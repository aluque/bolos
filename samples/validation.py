""" 
This is the 
"""
import numpy as np
import scipy.constants as co
from bolos import parser, solver, grid


def main():
    # Use a linear grid from 0 to 60 eV with 500 intervals.
    gr = grid.LinearGrid(0, 10, 200)

    # Initiate the solver instance
    bsolver = solver.BoltzmannSolver(gr)

    # Parse the cross-section file in BOSIG+ format and load it into the
    # solver.
    with open("itikawa-2009-O2.txt") as fp:
        bsolver.load_collisions(parser.parse(fp))

    # Set the conditions.  And initialize the solver
    T = 1000
    P = 101325
    ND = P / co.k / T
    bsolver.target['O2'].density = 1.0
    bsolver.kT = T * co.k / co.eV
    E = 1e5
    bsolver.EN = E / ND
    bsolver.init()

    # Start with Maxwell EEDF as initial guess.  Here we are starting with
    # with an electron temperature of 2 eV
    f0 = bsolver.maxwell(2.0)

    # Solve the Boltzmann equation with a tolerance rtol and maxn iterations.
    f1 = bsolver.converge(f0, maxn=200, rtol=1e-5)

    # Calculate the properties.
    print("mobility  = %.3f  1/m/V/s" % (bsolver.mobility(f1) / ND))
    print("diffusion  = %.3f  1/m/s" % (bsolver.diffusion(f1) / ND))
    print("average energy = %.3f  eV" % bsolver.mean_energy(f1))
    print("electron temperature = %.3f K" % bsolver.electron_temperature(f1))

if __name__ == '__main__':
    main()
