# -*- coding: iso-8859-1 -*-
from numpy import arctan
from numpy import pi
from numpy import gradient
from data_reader import Data
from scipy.interpolate import interp1d
import math
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline, BSpline
from scipy import interpolate
from utils import getGaussian, getInverseExponential, getDeepCopy, linInterpolArray1D

class Manipulator:
    """
    @ class information
    - This class can manipulate the density profile. There are several types of manipulation that you can each call
        with an scaling factor that will influces the strength of the manipulation.
    - CONSTRUCTOR: one has to pass the data set (see data_reader.py) set that also includes the original ne profile (=working profile).
    - NOTE: The manipulation functions are stateless -> they do not change the working profile (self.ne)
        but only return the manipulated profile
        If you want to combine multiple manipulations on the same profile, you have to make use of the setNe() method
        before each manipulation to update the working profile with your latest profile    

    @ methods to call
    - setTime:
        * specify time where you want to manipulate ne profile
        * defaut if you do not call it is t=3s
    - constShiftHorizontal:
        * constant shift of the whole graph in horizontal direction
        * gradient does not change
    - constShiftVertical:
        * constant shift of the whole graph in vertical direction
        * gradient does not change
    - verticalShift:
        * changes the steepness of both the pedestrial and the edge
        * gradient will increase and get a bit wider
    - horizontalShift:
        * can widen or shorten the length of the edge
        * the method will find the wendepunkt, cut the graph there
            and move the left part along the gradient axis
    """

    def __init__(self, data: Data):
        self.RHOS = 1 # rho seperatrix
        self.rho = np.array(data.rho)
        self.ne = np.array(data.ne)
        self.data = data
        # default values:
        self.setTime(3)
        self.setBoundariesAutomatically()

    def setNe(self, ne: list):
        self.ne = np.array(ne)

    def setTime(self, time: float):
        self.ne = self.data.getNeFromTime(time)

    def setBoundaries(self, rho1: float = None, rho2: float = None):
        # processes given boundaries (also if they are None)
        if rho1 and rho2:
            self.RHO1 = rho1
            self.RHO2 = rho2
            self.setRhoIndices()
        elif not rho1 and not rho2:
            self.setBoundariesAutomatically()

    def setRhoIndices(self):
        rho = self.rho
        self.idx1 = np.where(rho >= self.RHO1)[0][0]
        self.idx2 = np.where(rho >= self.RHO2)[0][0]
        self.idxS = np.where(rho >= self.RHOS)[0][0]
        self.idxDelta = self.idx2 - self.idx1

    def setBoundariesAutomatically(self):
        # this will set idx1, idx2 automatically by analyzing the density gradient
        THRESHHOLD = 0.1
        rho = self.rho
        idxS = np.where(rho >= self.RHOS)[0][0]
        grad = gradient(self.ne)
        verticalDelta = abs(grad[idxS] - grad[-1])
        delta = THRESHHOLD*verticalDelta
        # find idx1
        i = j = idxS
        while(i >= 0 and abs(grad[i]) > delta):
            self.idx1 = i
            i -= 1
        # find idx2
        while(j < len(grad) and abs(grad[j]) > delta):
            self.idx2 = j
            j += 1
        self.idxDelta = self.idx2 - self.idx1

    def getRhoIndices(self, rho2: float):
        idx1 = 0
        idx2 = np.where(self.rho >= rho2)[0][0]
        return idx1, idx2, idx2

    def verticalShift(self, scaleFactor=None, rho2: float=1) -> list:
        # Impact on steepness of whole graph but folded with arctan
        # so that profile stays quite flat at low rho and changes gradient at rho=1
        min, max = -1, 1
        if not scaleFactor or scaleFactor < min or scaleFactor > max:
            scaleFactor=self.getScaleFactor(min,max)
        idx1, idx2, idxDelta = self.getRhoIndices(rho2)
        ne_m = getDeepCopy(self.ne)
        x = np.linspace(5, 0, num=idxDelta)
        #order = (int(log10(ne_m[idx1])))
        #ne_m[idx1:idx2] += 10**order*scaleFactor*arctan(x) #(int(log10(ne_m[idx1])))
        ne_m[idx1:idx2] += ne_m[idx1:idx2] * arctan(x) * scaleFactor * 2 / pi
        return ne_m

    def horizontalShift(self, scaleFactor=None) -> list:
        # Shifts index of ne along the gradient vector
        min, max = -1, 1
        if not scaleFactor or scaleFactor < min or scaleFactor > max:
            scaleFactor=self.getScaleFactor(min,max)
        idx1, idx2, ne_m = self.idx1,self.idx2, getDeepCopy(self.ne)
        # compute wendepunkt and save as idxC
        idxC = self.idx1 + abs(gradient(ne_m[self.idx1:self.idx2])).argmax()
        grad = k = abs(gradient(ne_m)[idxC])            
        deltaX = abs(int(scaleFactor * 0.07 * abs(idxC)))
        if scaleFactor > 0:          
            newL = [None] * idxC
            for i, elem in reversed(list(enumerate(ne_m[: idxC]))):
                xNew = i - deltaX
                yNew = elem + k*deltaX
                if xNew >= 0 and xNew <= len(newL)-1:
                    newL[xNew] = yNew                
            ne_m[:idxC] = newL
        elif scaleFactor < 0:           
            newR = [None] * (idxC + abs(deltaX))
            for i, elem in reversed(list(enumerate(ne_m[: idxC]))):
                xNew = i + deltaX
                yNew = elem - k*deltaX
                if xNew >= 0 and xNew <= len(newR)-1:
                    newR[xNew] = yNew
            ne_m[:idxC+abs(deltaX):] = newR
        # interpolate none values
        nan_indices = np.isnan(ne_m)
        all_indices = np.arange(len(ne_m))
        interp_func = interp1d(all_indices[~nan_indices], ne_m[~nan_indices], kind='cubic', fill_value='extrapolate')
        ne_m[nan_indices] = interp_func(all_indices[nan_indices])
        return ne_m

    def constShiftHorizontal(self, scaleFactor=None):
        # Shifts whole function horizontally to left or right
        min, max, ne, length = -1, 1, getDeepCopy(self.ne), len(self.ne)
        if not scaleFactor or scaleFactor < min or scaleFactor > max:
            scaleFactor=self.getScaleFactor(min,max)
        l1, l2 = np.array([ne[0]] * length), np.array([ne[-1]] * length)
        new = np.concatenate((l1, ne, l2))
        constShift = int(length * scaleFactor)
        return new[length + constShift : 2*length + constShift ]

    def constShiftVertical(self, scaleFactor=None):
        min, max, ne, length = -1, 1, getDeepCopy(self.ne), len(self.ne)
        if not scaleFactor or scaleFactor < min or scaleFactor > max:
            scaleFactor=self.getScaleFactor(min,max)
        return [elem+ne[0]*scaleFactor for elem in ne]

    def getScaleFactor(self, min, max) -> float:
        while True:
            scaleFactor = float(input(f"Enter scale factor (between {min} and {max}): "))
            if scaleFactor >= min and scaleFactor <= max:
                return scaleFactor
            print(f"Please provide a number between {min} and {max}")

    def getNeDeepCopy(self):
        return [elem for elem in self.ne]
