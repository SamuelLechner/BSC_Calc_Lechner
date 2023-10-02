import numpy as np
import scipy.stats as stats
from scipy.interpolate import interp1d
import threading
import math

def linInterpolArray1D(a: list, linIntLvl: int) -> list:
    """
        This function returns a list of size specified in variable: self.linIntLvl
        This is mainly needed in order to have a better resolution in the manipulation process
        and other calculations.
        Note that it's only necessary to interplate the rho dependent values.
    """
    new_length = linIntLvl
    x = np.linspace(0, len(a) - 1, len(a))
    new_x = np.linspace(0, len(a) - 1, new_length)
    #result = np.interp(new_x, x, a)
    f = interp1d(x, a, kind='cubic')
    return f(new_x)


    return result

def linInterpolArray2D(a: list, linIntLvl: int) -> list:
    # First index must be rho!
    a = np.transpose(a)
    result = np.array([])
    for elem in a:
        result = np.vstack([result, linInterpolArray1D(elem, linIntLvl)]) if result.size else linInterpolArray1D(elem, linIntLvl)
    return np.transpose(result)

def getGaussian(newCut: list) -> list:
    mu = 0
    variance = 1
    sigma = np.sqrt(variance)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, len(newCut))
    y = [math.log(elem, 10) for elem in stats.norm.pdf(x, mu, sigma)]
    y = [(elem-y[0]) for elem in y]
    result = [(elem) / abs(np.max(y)) for elem in y]
    result = np.where(np.isnan(result),0,result) #replace NaN with 0
    return result

def getInverseExponential(newCut: list) -> list:
    x = np.linspace(0, 2, num=len(newCut))
    return 1/np.exp(x)

def getDeepCopy(a: list) -> list:
    return a.copy()

def mapTiToNewRho(ti: list, rhoOld: list, rhoNew: list) -> list:
    # Note this can only handle 1D ti array
    # Calcs index at which in rhoNew the rhoOld ends
    startIndex = np.where(rhoNew >= rhoOld[-1])[0][0] - 1
    # Calcs length for new, interpolated array
    newLength = rhoNew[:startIndex].size
    # Calculate the new indices for the interpolated array
    x = np.linspace(0, rhoNew[startIndex], newLength)
    xp = rhoOld
    # Calculate the amount of padding needed
    padding_length = rhoNew.size - newLength
    # init new array with shape(time, rho)
    tiNew = np.array([])
    # loop over all time
    for fp in ti:
        #print(fp)
        # Interpolate the array to the new length
        #temp = np.interp(x, xp, fp)
        f = interp1d(xp, fp, kind='cubic')
        temp = np.array(f(x))
        # Pad the array with the constant value
        temp = np.pad(temp, (0, padding_length), mode='constant', constant_values=temp[-1])
        tiNew = np.vstack([tiNew, temp]) if tiNew.size else temp
    # Transpose so that first index is rho
    tiNew = np.transpose(tiNew)
    return tiNew
