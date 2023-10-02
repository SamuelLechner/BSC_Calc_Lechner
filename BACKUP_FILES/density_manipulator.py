# -*- coding: iso-8859-1 -*-
from numpy import arctan
from numpy import pi
from numpy import gradient
from data_reader import Data
import math
import numpy as np
import pandas as pd
from utils import getGaussian,getDeepCopy

class Manipulator:
    """
    @ class information
    - This class can manipulate the density profile between two points RHO1 and RHO2 (refering to rho)

    @ parameters / (can bet set in constructor!)
    - RHO1, RHO2: will determine points in between manipulation takes place

    @ expected input values
    - rho is expected to be a 1-D array, no time dependency
    - ne is expected to be a 1-D array, dependency on rho, no time dependency
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

    def horizontalShift(self, scaleFactor=None):
        # Shifts index of ne in symetric interval around p where p is the wendepunkt of ne
        # shift value is not constant but is folded with gaussian distribution so that
        # greatest shift will take place at p
        min, max = -1, 1
        if not scaleFactor or scaleFactor < min or scaleFactor > max:
            scaleFactor=self.getScaleFactor(min,max)
        idx1, idx2, ne_m = self.idx1,self.idx2, getDeepCopy(self.ne)
        idxC = self.idx1+abs(gradient(ne_m[self.idx1:self.idx2])).argmax()
        #print("Wendepunkt berechnet bei: ", self.rho[idxC])
        idx2 = 2*idxC-idx1
        newCut = abs(idx2-idx1)*[None]
        # get gaussian curve which contains a value between 0 and 1
        y = getGaussian(newCut)
        # horizontal index shift in range(idx1, idx2)
        deltaHor = idx2-idxC #default
        for i,elem in enumerate(ne_m[idx1:idx2]):
            j = i + int(deltaHor * scaleFactor * y[i])
            if idx1+j >= idx1 and idx1+j <= idx2 and j<len(newCut):
                newCut[j] = elem
        # interpolate none entries
        newCut.insert(0,(ne_m[idx1]))
        newCut.append((ne_m[idx2]))
        series = pd.Series(newCut)
        interpolated = series.interpolate()
        newCutInt = interpolated.tolist()
        ne_m[idx1-1:idx2+1] = newCutInt
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
