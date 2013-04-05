import logging

import numpy as np

from process import Process, ScaledProcess, CombinedProcess, NullProcess


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

        self.combined = {}

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

        if process.kind == 'ELASTIC':
            f = self.mass_ratio or 1.0
            p = ScaledProcess(process, 2 * f)
            p.kind = 'WEIGHTED_ELASTIC'
            p.mass_ratio = None
            self.add_process(p)

        process.target = self

        logging.debug("Process %s added to target %s" 
                      % (str(process), str(self)))

    def set_energy_grid(self, eps):
        for key, proc in self.combined.iteritems():
            proc.set_energy_grid(eps)

        if hasattr(self, 'pair'):
            # It's a coposite target
            for t in self.pair:
                t.set_energy_grid(eps)

            return

        for key, plist in self.kind.iteritems():
            for proc in plist:
                proc.set_energy_grid(eps)
        

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


    def combined_process(self, kind):
        """ Combines all processes of a given kind into a single process. """
        if not self.kind[kind]:
            # We make sure that we always return something that can be managed
            logging.debug("No processes '%s' in target %s" % (kind, self.name))
            return NullProcess(self.name, kind)

        return reduce(CombinedProcess, self.kind[kind])
    

    def combine_all(self):
        """ Combines all kinds of processes into CombinedProcesses. """
        for key, item in self.kind.iteritems():
            proc = self.combined_process(key)
            setattr(self, 'all_%s' % key.lower(), proc)
            self.combined[key] = proc

        self.all_inelastic = reduce(CombinedProcess,
                                    [self.all_attachment, 
                                     self.all_excitation,
                                     self.all_ionization])

        self.all_all = CombinedProcess(self.all_elastic, self.all_inelastic)
        
        self.combined['INELASTIC'] = self.all_inelastic
        self.combined['ALL'] = self.all_all


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


class CombinedTarget(Target):
    """ A class of two combined targets. """
    def __init__(self, he, she):
        name = "/".join([he.name, she.name])
        super(CombinedTarget, self).__init__(name)

        self.density = he.density + she.density

        for key, item in self.kind.iteritems():
            for p in he.kind[key]:
                self.kind[key].append(
                    ScaledProcess(p, he.density / self.density))

            for p in she.kind[key]:
                self.kind[key].append(
                    ScaledProcess(p, she.density / self.density))
        
        self.combine_all()
        self.pair = [he, she]
