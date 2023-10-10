# -*- coding: iso-8859-1 -*-
from data_reader import Data
from plotter import Plot
from density_manipulator import Manipulator
from bootstrap_calculator import Bootstrap
from collisionality_calculator import Collisionality
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
        self.thesisRoutine()


    """
    routine for bachelor thesis
    """

    def thesisRoutine(self):
        # @ PARAMS final
        self.SHOT = 36165 #41493 #38550
        self.TIME = 5
        self.TIME2 = 6
        self.RHO1 = 0.9
        self.RHO2 = 1.05
        self.p1 = 0.85
        self.p2 = 1.1
        self.SF_HOR_SHIFT = -0.3
        self.SF_VER_SHIFT = 0.1
        self.SF_CONST_SHIFT = -0.07

        self.singleNe() #CHAPTER: Single ne profile
        self.singleBs() #CHAPTER: Computing single bs profile
        self.multipleShiftsNe() #CHAPTER: Introducing all manipulation functions
        self.multipleShiftsBs() #CHAPTER: Recomputing bs with manipulated ne functions

    def singleNe(self): # Loading the density profile
        data = Data(self.SHOT)
        p = Plot(data, mode='ne')
        p.setTime(self.TIME)
        ne = data.getNeFromTime(self.TIME)
        p.addNeOrig(ne)
        p.save(f'Plots/SingleNe/ne')
        #p.show()

    def singleBs(self):
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
        p.save(f'Plots/SingleBs/bs')
        #p.show()


    def multipleShiftsNe(self):
        self.multShiftShot = self.SHOT
        p1,p2 = self.p1, self.p2
        data = Data(shot=self.multShiftShot, lvl=200)
        ne = data.getNeFromTime(self.TIME)
        pCH = Plot(data, mode='ne')
        pCV = Plot(data, mode='ne')
        pH = Plot(data, mode='ne')
        pV = Plot(data, mode='ne')
        pCH.setParams(time=self.TIME, iterations=4, ne=ne, p1=p1, p2=p2)
        pCV.setParams(time=self.TIME, iterations=4, ne=ne, p1=p1, p2=p2)
        pH.setParams(time=self.TIME, iterations=4, ne=ne, p1=p1, p2=p2)
        pV.setParams(time=self.TIME, iterations=4, ne=ne, p1=p1, p2=p2)
        m = Manipulator(data)
        m.setTime(self.TIME)
        m.setBoundaries(self.RHO1, self.RHO2)
        valuesconstShiftHorizontal = [-0.02,-0.01,0.01,0.02]
        valuesconstShiftVertical = [-0.1,-0.05,0.05,0.1]
        valuesHorShift = [-0.2,-0.1,0.1,0.2]
        valuesVerShift = [0.2,0.4,0.6,0.8]
        for val in zip(valuesconstShiftHorizontal,valuesconstShiftVertical,valuesHorShift,valuesVerShift):
            ne_mCH = m.constShiftHorizontal(val[0])
            ne_mCV = m.constShiftVertical(val[1])
            ne_mH = m.horizontalShift(val[2])
            ne_mV = m.verticalShift(val[3],rho2=1.05)
            pCH.addNeMan(ne_mCH)
            pCV.addNeMan(ne_mCV)
            pH.addNeMan(ne_mH)
            pV.addNeMan(ne_mV)

        pCH.save('Plots/MultipleNe/constShiftHorizontal')
        pCV.save('Plots/MultipleNe/constShiftVertical')
        pH.save('Plots/MultipleNe/horizontalShift')
        pV.save('Plots/MultipleNe/verticalShift')
        #pCH.show()
        #pCV.show()
        #pH.show()
        #pV.show()

    def multipleShiftsBs(self):
        self.multShiftShot = self.SHOT
        p1,p2 = self.p1, self.p2
        data = Data(shot=self.multShiftShot, lvl=200)
        ne = data.getNeFromTime(self.TIME)
        pCH = Plot(data, mode='stacked')
        pCV = Plot(data, mode='stacked')
        pH = Plot(data, mode='stacked')
        pV = Plot(data, mode='stacked')
        pCH.setParams(time=self.TIME, iterations=4, ne=ne, p1=p1, p2=p2)
        pCV.setParams(time=self.TIME, iterations=4, ne=ne, p1=p1, p2=p2)
        pH.setParams(time=self.TIME, iterations=4, ne=ne, p1=p1, p2=p2)
        pV.setParams(time=self.TIME, iterations=4, ne=ne, p1=p1, p2=p2)
        m = Manipulator(data)
        b = Bootstrap(data)
        c = Collisionality(self.SHOT)
        m.setTime(self.TIME)
        m.setBoundaries(self.RHO1, self.RHO2)
        valuesconstShiftHorizontal = [-0.02,-0.01,0.01,0.02]
        valuesconstShiftVertical = [-0.1,-0.05,0.05,0.1]
        valuesHorShift = [-0.2,-0.1,0.1,0.2]
        valuesVerShift = [0.2,0.4,0.6,0.8]
        cOrig = c.get_collisionalty(shot = self.SHOT, t = self.TIME)
        bsOrig = b.getBsCurrent(t1 = self.TIME)
        temp = data.getTempFromTime(t = self.TIME)
        pCH.addBasics(bs = bsOrig, c = cOrig, temp = temp)
        pCV.addBasics(bs = bsOrig, c = cOrig, temp = temp)
        pH.addBasics(bs = bsOrig, c = cOrig, temp = temp)
        pV.addBasics(bs = bsOrig, c = cOrig, temp = temp)
        for val in zip(valuesconstShiftHorizontal, valuesconstShiftVertical, valuesHorShift, valuesVerShift):
            # const horizontal
            ne_m = m.constShiftHorizontal(val[0])
            bs_m = b.getBsCurrent(t1 = self.TIME, ne_m = ne_m)
            c_m = c.get_collisionalty(shot = self.SHOT, t = self.TIME, ne = ne_m)
            pCH.addColliMan(c = c_m)
            pCH.addNeMan(ne_m)
            pCH.addBsMan(bs_m)
            # const vertical
            ne_m = m.constShiftVertical(val[1])
            bs_m = b.getBsCurrent(t1 = self.TIME, ne_m = ne_m)
            c_m = c.get_collisionalty(shot = self.SHOT, t = self.TIME, ne = ne_m)
            pCV.addColliMan(c = c_m)
            pCV.addNeMan(ne_m)
            pCV.addBsMan(bs_m)
            # horizontal
            ne_m = m.horizontalShift(val[2])
            bs_m = b.getBsCurrent(t1 = self.TIME, ne_m = ne_m)
            c_m = c.get_collisionalty(shot = self.SHOT, t = self.TIME, ne = ne_m)
            pH.addColliMan(c = c_m)
            pH.addNeMan(ne_m)
            pH.addBsMan(bs_m)
            # horizontal
            ne_m = m.verticalShift(val[3])
            bs_m = b.getBsCurrent(t1 = self.TIME, ne_m = ne_m)
            c_m = c.get_collisionalty(shot = self.SHOT, t = self.TIME, ne = ne_m)
            pV.addColliMan(c = c_m)
            pV.addNeMan(ne_m)
            pV.addBsMan(bs_m)

        pCH.save('Plots/MultipleBs/constShiftHorizontal')
        pCV.save('Plots/MultipleBs/constShiftVertical')
        pH.save('Plots/MultipleBs/horizontalShift')
        pV.save('Plots/MultipleBs/verticalShift')
        #pCH.show()
        #pCV.show()
        #pH.show()
        #pV.show()

        


if __name__ == '__main__':
    main = Main()