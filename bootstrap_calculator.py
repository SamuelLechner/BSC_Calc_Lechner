# -*- coding: iso-8859-1 -*-
import numpy as np
from data_reader import Data
from numpy import sqrt
from numpy import gradient
from scipy.interpolate import interp1d as ip1d
from utils import *
import scipy.constants as constants

class Bootstrap:
    """
    @ class information
    - This class can compute the bootstrap bs_current
    - CONSTRUCTOR: takes the data set as an argument (see data_reader.py)
    
    @ methods to call
    - getBsCurrent:
        - returns the calculated bootstrap current
        - ARGUMENTS:
            1) You pass t1 and t2: You get the average bs current in that time interval
            2) You pass only t1: You get the the stationary bs current at t=t1
            3) You can pass a ne list in case you want to apply calc on a manipulted ne profile
    """

    def __init__(self, data: Data):
        self.data = data
        self.bSet = True

    def calcBs(self, rho):
        b = None
        if self.bSet:
            try:
                b = lambda rho: self.data.b[rho]
            except AttributeError:
                # This happens if we have no static b defined (because user forgot to init it in the data class so we need to calc it for each rho)
                print(f"Attention, you might face a long calculation time since you have not used 'data.setbFromTime(momentInSec)' method.\
                This would help executing intense calculation routines only one time instead of multiple times.") if self.bSet else None
                self.bSet = False
                b = lambda rho: self.data.get_b(data.rho[rho])
        else:
            b = lambda rho: self.data.get_b(data.rho[rho])
        return (-sqrt(b(rho)) * self.maj_rad * self.data.bt) / (self.data.b_0) * (2.44*(self.te[rho]+self.ti[rho])*gradient(self.ne)[rho] + 0.69*self.ne[rho]*gradient(self.te)[rho] - 0.42*self.ne[rho]*gradient(self.ti)[rho])

    def getBsCurrent(self, t1: float, t2: float = None, ne_m: list = None) -> list:
        # ne_m provides option to pass manipulated ne list
        t2 = t1 if not t2 else t2
        data = self.data
        # constants
        boltzmann = constants.Boltzmann
        e = constants.elementary_charge
        eV = 1 / e
        # for time frame
        idxDelta = data.getIdxDeltaFromTimeframe(t1,t2)
        timePoints = np.linspace(t1, t2, num=idxDelta) # num=idxDelta;   [1] if t1=t2
        bsTime = np.array([])
        for time in timePoints:
            data.setData(time)
            data.setbFromTime(time)
            if ne_m is None:
                self.ne = data.ne[:, data.idx_ida]
            else:
                self.ne = np.array(ne_m)

            self.te = data.te[:,data.idx_ida]
            self.ti = data.ti[:,data.idx_idi]
            self.maj_rad = data.maj_rad[data.idx_idg] # maj rad is time dependent

            bsRho = np.array([])
            for rho in range(len(data.rho)):
                elem = self.calcBs(rho) * e  / 1000 # return in [mega ampere / m^2]
                bsRho = np.append(bsRho, np.absolute(elem))
            bsTime = np.vstack([bsTime, bsRho]) if bsTime.size else bsRho
        bsTime = np.transpose(bsTime)
        return np.average(bsTime, axis = 1) if bsTime.ndim >= 2 else bsTime
