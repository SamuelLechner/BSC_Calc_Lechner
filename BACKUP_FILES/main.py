# -*- coding: iso-8859-1 -*-
from data_reader import Data
from plotter import Plot
from density_manipulator import Manipulator
from bootstrap_calculator import Bootstrap
import plotter
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

class Main:
    """
    @ class information
    - This is the main function.
    - It's only necessary to launch this file and everything else will be handled/executed from here.
    """
    def __init__(self):
        #self.testDensityManipulation()
        #self.testBootstrapCalculation()
        #self.testBootstrapCompare()
        self.thesisRoutine()

    """
    routine for bachelor thesis
    """

    def thesisRoutine(self):
        # @ PARAMS
        self.SHOT = 36165 #41493 # 38550
        self.TIME = 5
        self.TIME2 = 6
        self.RHO1 = 0.9
        self.RHO2 = 1.05
        self.SF_HOR_SHIFT = -0.3
        self.SF_VER_SHIFT = 0.1
        self.SF_CONST_SHIFT = -0.07

        #self.loadingTheDensityProfile()
        #self.computingTheBootstrapCurrent()
        #self.manipulatingDensityProfiles()
        #self.computingBootstrapUsingManipulatedNe()
        #self.comparingBootstrapModels()
        #self.multipleShiftsNe()
        #self.multipleShiftsBs()
        self.singleBootstrapMan()

    def loadingTheDensityProfile(self): # Loading the density profile
        data = Data(self.SHOT)
        p = Plot(data, mode='ne')
        p.setTime(self.TIME)
        p.setPlottingBoundaries(0.8,1.1)
        ne = data.getNeFromTime(self.TIME)
        p.addNeOrig(ne)
        p.save(f'ne')
        #p.show()

    def computingTheBootstrapCurrent(self):
        data = Data(self.SHOT)
        m = Manipulator(data)
        bs = Bootstrap(data)
        p = Plot(data, mode='bs')
        m.setTime(self.TIME)
        p.setTime(self.TIME)
        ne = data.getNeFromTime(self.TIME)
        bsAvg = bs.getBsCurrent(self.TIME)
        p.addBsOrig(bsAvg)
        p.addNeOrig(ne)
        p.save(f'bs')
        #p.show()

    def comparingBootstrapModels(self):
        # init
        data = Data(self.SHOT)
        m = Manipulator(data)
        bs = Bootstrap(data)
        bs_p = bs.getBsCurrent(self.TIME)
        p = Plot(data, mode='bs')
        # Get bs values from Redl method (calc by M Dunne)
        yfilename = '/afs/ipp/home/s/slech/Documents/BSC_Calc_Lechner/Compare_Models_with_Shot38550/dunne/y_val'
        xfilename = '/afs/ipp/home/s/slech/Documents/BSC_Calc_Lechner/Compare_Models_with_Shot38550/dunne/x_val'
        rho_r, bs_r = np.array([]), np.array([])
        with open(xfilename) as f:
            for line in f:
                values = line.strip().split()
                for val in values:
                    rho_r = np.append(rho_r, float(val))
        with open(yfilename) as f:
            for line in f:
                values = line.strip().split()
                for val in values:
                    if val=='NaN' or val=='-NaN':
                        val=0
                    bs_r = np.append(bs_r, float(val))
        # Plot
        p.addBsRedl(rho_r, bs_r)
        p.addBsPeeters(bs_p)
        p.show()

    def manipulatingDensityProfiles(self):
        data = Data(self.SHOT)
        p = Plot(data, mode='ne')
        p.setIterations(2)
        p.setTime(self.TIME)
        ne = data.getNeFromTime(self.TIME)
        m = Manipulator(data)
        m.setTime(self.TIME)
        m.setBoundaries(0.9, 1.05)
        ne_m1 = m.horizontalShift(0.15)
        p.addNeMan(ne_m1,plotGrad=True)
        ne_m2 = m.horizontalShift(0.2)
        p.addNeMan(ne_m2,plotGrad=True)
        #ne_m = m.horizontalShift(self.SF_HOR_SHIFT)
        #m.setNe(ne_m)
        #ne_m = m.verticalShift(self.SF_VER_SHIFT)
        p.addNeOrig(ne)
        #p.addNeMan(ne_m)
        p.show()

    def multipleShiftsNe(self):
        p1,p2 = 0.85,1.1
        self.multShiftShot = self.SHOT
        data = Data(self.multShiftShot)
        ne = data.getNeFromTime(self.TIME)
        pC = Plot(data, mode='ne')
        pH = Plot(data, mode='ne')
        pV = Plot(data, mode='ne')
        pC.setParams(time=self.TIME, iterations=4, ne=ne, p1=p1, p2=p2)
        pH.setParams(time=self.TIME, iterations=4, ne=ne, p1=p1, p2=p2)
        pV.setParams(time=self.TIME, iterations=4, ne=ne, p1=p1, p2=p2)
        m = Manipulator(data)
        m.setTime(self.TIME)
        m.setBoundaries(self.RHO1, self.RHO2)
        valuesconstShiftHorizontal = [-0.02,-0.01,0.01,0.02]
        valuesHorShift = [-0.2,-0.1,0.1,0.2]
        valuesVerShift = [0.2,0.4,0.6,0.8]
        for val in zip(valuesconstShiftHorizontal,valuesHorShift,valuesVerShift):
            ne_mC = m.constShiftHorizontal(val[0])
            ne_mH = m.horizontalShift(val[1])
            ne_mV = m.verticalShift(val[2],rho2=1.05)
            pC.addNeMan(ne_mC)
            pH.addNeMan(ne_mH)
            pV.addNeMan(ne_mV)

        pC.save('MultipleShiftsNe/constShiftHorizontal')
        pH.save('MultipleShiftsNe/horizontalShift')
        pV.save('MultipleShiftsNe/verticalShift')
        # pC.show()
        # pH.show()
        # pV.show()

    def multipleShiftsBs(self):
        p1,p2 = 0.85,1.1
        self.multShiftShot = self.SHOT
        data = Data(self.multShiftShot)
        ne = data.getNeFromTime(self.TIME)
        pC = Plot(data, mode='stacked')
        pH = Plot(data, mode='stacked')
        pV = Plot(data, mode='stacked')
        b = Bootstrap(data)
        bs = b.getBsCurrent(t1=self.TIME)
        pC.addBsOrig(bs)
        pH.addBsOrig(bs)
        pV.addBsOrig(bs)
        pC.setParams(time=self.TIME, iterations=4, ne=ne, p1=p1, p2=p2)
        pH.setParams(time=self.TIME, iterations=4, ne=ne, p1=p1, p2=p2)
        pV.setParams(time=self.TIME, iterations=4, ne=ne, p1=p1, p2=p2)
        m = Manipulator(data)
        m.setTime(self.TIME)
        m.setBoundaries(self.RHO1, self.RHO2)
        valuesconstShiftHorizontal = [-0.02,-0.01,0.01,0.02]
        valuesHorShift = [-0.2,-0.1,0.1,0.2]
        valuesVerShift = [0.2,0.4,0.6,0.8]
        for val in zip(valuesconstShiftHorizontal,valuesHorShift,valuesVerShift):
            ne_mC = m.constShiftHorizontal(val[0])
            bs_mC = b.getBsCurrent(t1=self.TIME, ne_m=ne_mC)
            ne_mH = m.horizontalShift(val[1])
            bs_mH = b.getBsCurrent(t1=self.TIME, ne_m=ne_mH)
            ne_mV = m.verticalShift(val[2],rho2=1.05)
            bs_mV = b.getBsCurrent(t1=self.TIME, ne_m=ne_mV)
            pC.addNeMan(ne_mC)
            pC.addBsMan(bs_mC)
            pH.addNeMan(ne_mH)
            pH.addBsMan(bs_mH)
            pV.addNeMan(ne_mV)
            pV.addBsMan(bs_mV)

        pC.save('MultipleShiftsBs/constShiftHorizontal')
        pH.save('MultipleShiftsBs/horizontalShift')
        pV.save('MultipleShiftsBs/verticalShift')
        # pC.show()
        # pH.show()
        # pV.show()

    def singleBootstrapMan(self):
        p1,p2 = 0, 1.2
        self.multShiftShot = self.SHOT
        data = Data(shot=self.multShiftShot,lvl=0)
        data.setbFromTime(self.TIME)
        ne, *Temp, bCoeff = data.getDataFromTime(self.TIME)
        p = Plot(data, mode='stacked')
        p.setParams(time=self.TIME, iterations=4, ne=ne, p1=p1, p2=p2)
        b = Bootstrap(data)
        bs = b.getBsCurrent(t1=self.TIME)
        p.addBasics(bs, Temp, bCoeff)
        m = Manipulator(data)
        m.setTime(self.TIME)
        m.setBoundaries(0.9, 1.1)
        valuesconstShiftHorizontal = [0.2,0.4,0.6,0.8]
        for val in valuesconstShiftHorizontal:
            ne_m = m.verticalShift(val,rho2=1.05)
            bs_m = b.getBsCurrent(t1=self.TIME, ne_m=ne_m)
            p.addNeMan(ne_m)
            p.addBsMan(bs_m)

        p.show()

if __name__ == '__main__':
    main = Main()
