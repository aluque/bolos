from collections import defaultdict
import logging

import numpy as np

from .process import Process, NullProcess


class Target(object):
    """ A class to contain all information related to one target. """

    def __init__(self, name):
        """ Initializes an instance of target named name."""
        self.name = name
        self.mass_ratio = None
        self.density = 0.0

        # Lists of all processes pertaining this target
        self.elastic             = []
        self.effective           = []
        self.attachment          = []
        self.ionization          = []
        self.excitation          = []
        self.weighted_elastic    = []

        self.kind = {'ELASTIC': self.elastic,
                     'EFFECTIVE': self.effective,
                     'MOMENTUM': self.effective,
                     'ATTACHMENT': self.attachment,
                     'IONIZATION': self.ionization,
                     'EXCITATION': self.excitation,
                     'WEIGHTED_ELASTIC': self.weighted_elastic}

        self.by_product = defaultdict(list)

        logging.debug("Target %s created." % str(self))

    def add_process(self, process):
        kind = self.kind[process.kind]
        kind.append(process)

        if process.mass_ratio is not None:
            logging.debug("Mass ratio (=%g) for %s" 
                          % (process.mass_ratio, str(self)))

            if (self.mass_ratio is not None 
                and self.mass_ratio != process.mass_ratio):
                raise ValueError("More than one mass ratio for target '%s'"
                                 % self.name)

            self.mass_ratio = process.mass_ratio

        process.target = self

        self.by_product[process.product].append(process)

        logging.debug("Process %s added to target %s" 
                      % (str(process), str(self)))

        

    def ensure_elastic(self):
        """ Makes sure that the process has an elastic cross-section.
        If the user has specified an effective cross-section, we remove
        all the other cross-sections from it. """
        if self.elastic and self.effective:
            raise ValueError("In target '%s': EFFECTIVE/MOMENTUM and ELASTIC"
                             "cross-sections are incompatible." % self)

        if self.elastic:
            return

        if len(self.effective) > 1:
            raise ValueError("In target '%s': Can't handle more that 1 "
                             "EFFECTIVE/MOMENTUM for a given target" % self)
            
        if not self.effective:
            logging.warning("Target %s has no ELASTIC or EFFECTIVE "
                            "cross sections" % str(self))
            return

        newdata = self.effective[0].data.copy()
        for p in self.inelastic:
            newdata[:, 1] -= p.interp(newdata[:, 0])

        if np.amin(newdata[:, 1]) < 0:
            logging.warning('After substracting INELASTIC from EFFECTIVE, '
                            'target %s has negative cross-section.'
                            % self.name)
            logging.warning('Setting as max(0, ...)')
            newdata[:, 1] = np.where(newdata[:, 1] > 0, newdata[:, 1], 0)


        newelastic = Process(target=self.name, kind='ELASTIC',
                             data=newdata,
                             mass_ratio=self.effective[0].mass_ratio,
                             comment="Calculated from EFFECTIVE cross sections")

        logging.debug("EFFECTIVE -> ELASTIC for target %s" % str(self))
        self.add_process(newelastic)

        # Remove the EFFECTIVE processes.
        self.effective = []


    @property
    def inelastic(self):
        """ An useful abbreviation. """
        return (self.attachment + self.ionization + self.excitation)

    @property
    def everything(self):
        """ A list with ALL processes.  We do not use all as a name
        to avoid confusion with the python function."""
        return (self.elastic + self.attachment 
                + self.ionization + self.excitation)

    
    def __repr__(self):
        return "Target(%s)" % repr(self.name)

    def __str__(self):
        return self.name

