""" Parse BOLSIG+ compatible files with cross-sections. 
    
"""

import sys
import re
import numpy as np

def parse(fp):
    processes = []
    for line in fp:
        try:
            key = line.strip()
            fread = KEYWORDS[key]
            d = fread(fp)
            d['type'] = key
            processes.append(d)
            
        except KeyError:
            pass

    return processes


# BOLSIG+'s user guide saye that the separators must consist of at least five dashes
RE_SEP = re.compile("-----+")
def read_until_sep(fp):
    """ Reads lines from fp until a we find a separator line. """
    lines = []
    for line in fp:
        if RE_SEP.match(line.strip()):
            break
        lines.append(line.strip())

    return lines


def read_block(fp, has_arg=True):
    """ Reads data of a process, contained in a block. 
    has_arg indicates wether we have to read an argument line"""
    target = fp.next().strip()
    if has_arg:
        arg = fp.next().strip()
    else:
        arg = None

    comment = "\n".join(read_until_sep(fp))

    #data = "\n".join(read_until_sep(fp))
    data = np.loadtxt(read_until_sep(fp)).tolist()

    return target, arg, comment, data

#
# Specialized funcion for each keyword. They all return dictionaries with the
# relevant attibutes.
# 
def read_momentum(fp):
    """ Reads a MOMENTUM or EFFECTIVE block. """
    target, arg, comment, data = read_block(fp, has_arg=True)
    ratio = float(arg.split()[0])
    d = dict(target=target,
             ratio=ratio,
             comment=comment,
             data=data)

    return d

RE_ARROW = re.compile('<?->')    
def read_excitation(fp):
    """ Reads an EXCITATION or IONIZATION block. """
    target, arg, comment, data = read_block(fp, has_arg=True)
    lhs, rhs = [s.strip() for s in RE_ARROW.split(target)]

    d = dict(target=lhs,
             product=rhs,
             comment=comment,
             data=data)

    if '<->' in target.split():
        threshold, weight_ratio = float(arg.split()[0]), float(arg.split()[1])
        d['weight_ratio'] = weight_ratio
    else:
        threshold = float(arg.split()[0])

    d['threshold'] = threshold
    return d


def read_attachment(fp):
    """ Reads an ATTACHMENT block. """
    target, arg, comment, data = read_block(fp, has_arg=False)

    d = dict(target=target,
             comment=comment,
             data=data)

    return d


KEYWORDS = {"MOMENTUM": read_momentum, 
            "ELASTIC": read_momentum, 
            "EFFECTIVE": read_momentum,
            "EXCITATION": read_excitation,
            "IONIZATION": read_excitation,
            "ATTACHMENT": read_attachment}



def main():
    import json
    import yaml

    with open(sys.argv[1]) as fp:
        processes = parse(fp)

    with open("crosssect.json", "w") as fp:
        fp.write(json.dumps(processes, indent=2))

    with open("crosssect.yaml", "w") as fp:
        yaml.dump(processes, fp)


if __name__ == '__main__':
    main()
