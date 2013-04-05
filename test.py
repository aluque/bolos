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

    bsolver = solver.BoltzmannSolver(0, 100.0)
    bsolver.load_collisions(processes)

    bsolver.target['N2'].density = 0.8
    bsolver.target['O2'].density = 0.2

    bsolver.EN = 120 * solver.TOWNSEND
    bsolver.kT = 300 * co.k / co.eV

    bsolver.init()
    pexc = bsolver.target['N2'].combined_process('EXCITATION')


    
    print bsolver.total.name
    p = bsolver.total
    #p = bsolver.target['N2']

    pylab.subplot(111)
    p.all_all.plot(pylab.gca(), '-o', lw=1.8, c='b')
    p.all_ionization.plot(pylab.gca(), '-o', lw=1.8, c='k')
    p.all_excitation.plot(pylab.gca(), '-o', lw=1.8, c='#aa77aa')
    p.all_inelastic.plot(pylab.gca(), '-o', lw=1.8, c='g')
    p.all_elastic.plot(pylab.gca(), '-o', lw=1.8, c='r')

    pylab.show()

    

if __name__ == '__main__':
    main()
