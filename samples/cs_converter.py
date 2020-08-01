import numpy as np
import scipy.constants as co
from bolos import parser, solver, grid
try:
    import ruamel_yaml as yaml
except ImportError:
    from ruamel import yaml


def main():

    # Parse the cross-section file in BOSIG+ format and load it into the
    # solver.
    with open("itikawa-2009-O2.txt") as fp:
        processes = parser.parse(fp)

    data = dict()

    for cs in processes:
        cs["kind"] = cs["kind"].lower()

    data['cross_section'] = processes

    outfile = open("oxygen_cross_sections.yaml", "w")
    yaml.dump(data, outfile)

if __name__ == '__main__':
    main()
