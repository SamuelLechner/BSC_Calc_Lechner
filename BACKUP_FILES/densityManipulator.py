# -*- coding: iso-8859-1 -*-
import numpy as np
from numpy import sin
from numpy import pi
from numpy import gradient
import matplotlib.pyplot as plt

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

    def __init__(self, rho: list, ne: list):
        rho = np.array(rho)
        rho = np.array(rho)
        ne = np.array(ne)
        self.RHO1 = 0.9
        self.RHO2 = 1.1
        self.RHOS = 1 # rho seperatrix
        self.ne_m = ne # only work on manipulated ne array, don't change original input
        self.setRhoIndices(rho)

    def setRhoIndices(self, rho: list):
        self.idx1 = np.where(rho >= self.RHO1)[0]
        self.idx2 = np.where(rho >= self.RHO1)[0]
        self.idxS = np.where(rho >= self.RHOS)[0]
        self.idxDelta = self.idx2 - self.idx1

    def setBoundariesAutomatically(self):
        # this will set idx1, idx2 automatically by analyzing the density gradient
        THRESHHOLD = 0.1
        grad = gradient(self.ne_m)
        verticalDelta = grad[self.idxS] - grad[-1]
        # find idx1
        i = j = self.idxS
        while(i >= 0 and grad[i] > THRESHHOLD*verticalDelta):
            self.idx1 = i
            i -= 1
        # find idx2
        while(j < len(grad) and grad[j] > THRESHHOLD*verticalDelta):
            self.idx2 = j
            j += 1



    def verticalShift(self):
        while True:
            scaleFactor = float(input("Enter scaling factor (between 0 and 1): "))
            if scaleFactor > 0 and scaleFactor <= 1:
                break
            print("Scaling factor is not between 0 and 1")

        idx1, idx2, ne_m = self.idx1, self.idx2, self.ne_m
        idxDelta = manipulator.idxDelta
        idxHalf = idxDelta // 2
        valueDelta = abs(ne_m[idx1]-ne_m[idx2])
        x = np.linspace(-idxHalf, idxHalf, num=idxDelta)
        sinShift = sin(pi * (idxHalf - abs(x)) / idxDelta)
        for i in range(idx1, idx2):
            if valueDelta-ne_m[i] < 0.5 * valueDelta:
                verticalDelta = abs(ne_m[idx1] - ne_m[i])
            else:
                verticalDelta = -abs(ne_m[idx2] - ne_m[i])
            ne_m[i] += sinShift[i-idx1] * verticalDelta * scaleFactor
        self.ne_m = ne_m


if __name__ == '__main__':
    # test case
    arr = [5.16, 5.13, 5.1, 5.07, 5.04, 5.01, 4.99, 4.96, 4.94, 4.93, 4.91, 4.9, 4.89, 4.89, 4.89, 4.89, 4.89, 4.89, 4.89, 4.89, 4.88, 4.87, 4.86, 4.86, 4.86, 4.79, 4.59, 4.21, 3.68, 3.04, 2.38, 1.82, 1.43, 1.19, 1.06, 0.96, 0.85, 0.72, 0.6, 0.5, 0.42, 0.35, 0.31, 0.28, 0.25, 0.23, 0.21, 0.19, 0.17, 0.15, 0.13, 0.11, 0.1, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01]
    manipulator = Manipulator(arr, arr)
    manipulator.idx1 = 22
    manipulator.idx2 = 50
    manipulator.idxDelta = manipulator.idx2-manipulator.idx1
    manipulator.verticalShift()

    plt.plot(arr, color="red", label='original')
    plt.plot(gradient(arr), color="lightcoral", label='gradient original')

    plt.plot(manipulator.ne_m, color="blue", label='manipulated')
    plt.plot(gradient(manipulator.ne_m), color="lightblue", label='gradient manipulated')
    plt.legend()
    plt.show()
    # test end
