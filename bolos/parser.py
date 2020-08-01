""" This module contains the code required to parse BOLSIG+-compatible files.
To make the code re-usabe in other projects it is independent from the rest of
the BOLOS code.

Most user would only use the method :func:`parse` in this module, which is 
documented below.
    
"""

import sys
import re
import numpy as np
import logging


def parse(fp):
    """ Parses a BOLSIG+ cross-sections file.  

    Parameters
    ----------
    fp : file-like
       A file object pointing to a Bolsig+-compatible cross-sections file.

    Returns
    -------
    processes : list of dictionaries
       A list with all processes, in dictionary form, included in the file.

    Note
    ----
    This function does not return :class:`process.Process` instances so that
    the parser is independent of the rest of the code and can be re-used in
    other projects.  If you want to convert a process in dictionary form `d` to
    a :class:`process.Process` instance, use

    >>> process = process.Process(**d)

    """
    processes = []
    for line in fp:
        try:
            key = line.strip()
            fread = KEYWORDS[key]

            # If the key is not found, we do not reach this line.
            logging.debug("New process of type '%s'" % key)

            d = fread(fp)
            d['kind'] = key
            processes.append(d)
            
        except KeyError:
            pass

    logging.info("Parsing complete. %d processes read." % len(processes))

    return processes


# BOLSIG+'s user guide saye that the separators must consist of at least five dashes
RE_SEP = re.compile("-----+")
def _read_until_sep(fp):
    """ Reads lines from fp until a we find a separator line. """
    lines = []
    for line in fp:
        if RE_SEP.match(line.strip()):
            break
        lines.append(line.strip())

    return lines


def _read_block(fp, has_arg=True):
    """ Reads data of a process, contained in a block. 
    has_arg indicates wether we have to read an argument line"""
    target = fp.readline().strip()
    if has_arg:
        arg = fp.readline().strip()
    else:
        arg = None

    comment = "\n".join(_read_until_sep(fp))

    logging.debug("Read process '%s'" % target)
    data = np.loadtxt(_read_until_sep(fp)).tolist()

    return target, arg, comment, data

#
# Specialized funcion for each keyword. They all return dictionaries with the
# relevant attibutes.
# 
def _read_momentum(fp):
    """ Reads a MOMENTUM or EFFECTIVE block. """
    target, arg, comment, data = _read_block(fp, has_arg=True)
    mass_ratio = float(arg.split()[0])
    d = dict(target=target,
             mass_ratio=mass_ratio,
             comment=comment,
             data=data)

    return d

RE_ARROW = re.compile('<?->')    
def _read_excitation(fp):
    """ Reads an EXCITATION or IONIZATION block. """
    target, arg, comment, data = _read_block(fp, has_arg=True)
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


def _read_attachment(fp):
    """ Reads an ATTACHMENT block. """
    target, arg, comment, data = _read_block(fp, has_arg=False)

    d = dict(comment=comment,
             data=data,
             threshold=0.0)
    lr = [s.strip() for s in RE_ARROW.split(target)]

    if len(lr) == 2:
        d['target'] = lr[0]
        d['product'] = lr[1]
    else:
        d['target'] = target

    return d


KEYWORDS = {"MOMENTUM": _read_momentum, 
            "ELASTIC": _read_momentum, 
            "EFFECTIVE": _read_momentum,
            "EXCITATION": _read_excitation,
            "IONIZATION": _read_excitation,
            "ATTACHMENT": _read_attachment}
