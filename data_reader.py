# -*- coding: iso-8859-1 -*-
import threading
import asyncio
import aug_sfutils as sf
import numpy as np
from utils import *
from scipy.interpolate import interp1d as ip1d

class Data:
    """
    @ class information
    - This class will read the data from the servers
    - Only variable to pass is the shot number
    - Values can be accessed with getter functions
    """

    def __init__(self, shot: int, lvl: int = None):
        self.shot = shot
        self.exp = 'lrado'
        self.startUp(lvl)

    def startUp(self, lvl):
        print(f"Reading data from server.")
        if not self.readDataFromServer():
            print(f'Error at reading Data from Server. Please try another shot.')
        else:
            self.linIntLvl = len(self.rho) if not lvl or lvl <= len(self.rho) else lvl
            # lvl ist the amout of datapoints one wants to have after interpolation, 300 defaults
            # usually there are 200 datapoints coming out of the asdex upgrade system at least for rho
            print(f'Data reading from server successful. Continue with interpolation (lvl = {self.linIntLvl}), this might take a while.')
            if self.interpolateDataCollection():
                print(f'Interpolation successful. Continue with further tasks.')
            else:
                print(f'Error at interpolation.')

    def readDataFromServer(self) -> bool:
        names = ["AUGD","LRADO", "GHARR"]
        idg,ida,idi,equ = None, None, None, None
        for name in names:
            try:
                equ = sf.EQU(self.shot, diag="IDE", exp=name)
                self.b = equ.get_profile("Bave")
                self.time_equ = equ.time
                break
            except AttributeError:
                continue
        if not equ:
            print(f'EQU cannot be read.')
            return False

        ida = sf.SFREAD(self.shot, 'ida', exp='AUGD')
        idg = sf.SFREAD(self.shot, 'idg', exp=self.exp)
        if not idg('TIMEF'):
            idg = sf.SFREAD(self.shot, 'idg', exp='AUGD')
        idi = sf.SFREAD(self.shot, 'idi', exp='AUGD')  # on idi there is only AUGD possible
        mai = sf.SFREAD("MAI", self.shot)

        self.equ = equ
        self.time_ida = ida('time')  # time for ida
        self.time_idi = idi('time')  # time for idi
        self.time_idg = idg('TIMEF') # time for idg
        self.time_mai = mai("T-MAG-1") # time for mai
        self.ne = ida('ne')  # electon density
        self.rho = ida('rhop')[:,0]  # radius, rho is constant over time
        self.te = ida('Te') # electron temperature
        self.ti = mapTiToNewRho(ti=idi('Ti'), rhoOld=idi('rp_Ti')[:, 0], rhoNew=self.rho) # ion temperature
        self.maj_rad = idg('Rmag')  # major plasma radius
        self.Bt = mai("BTF")

        return True

    def interpolateDataCollection(self):
        # TODO here one could apply multithreading for performance improvement
        linIntLvl, threads = self.linIntLvl, []
        if linIntLvl == self.rho.size:
            print("No interpolation needed.")
            return True
        self.ti = linInterpolArray2D(self.ti, linIntLvl)
        self.ne = linInterpolArray2D(self.ne, linIntLvl)
        self.te = linInterpolArray2D(self.te, linIntLvl)
        self.rho = linInterpolArray1D(self.rho, linIntLvl)
        return True


    def get_b(self, rho):
        # this function returns b for given rho
        Rin, zin = sf.rho2rz(eqm = self.equ, rho_in = rho, t_in = self.moment, coord_in = 'rho_pol', all_lines = False )
        Rin, zin = Rin[0][0], zin[0][0]
        br, bz, bt = sf.rz2brzt(eqm = self.equ, r_in = Rin, z_in = zin, t_in = 4)
        br, bz, bt = br[0], bz[0], bt[0]
        bp = np.hypot(br, bz)
        blist = np.hypot(bp, bt)
        b_max, b_min = np.max(blist), np.min(blist)
        b = (b_max - b_min) / (b_max + b_min)
        return b # note that b is 0 for low rhos

    def setbFromTime(self, t: float):
        self.moment = t
        self.b = [self.get_b(rho) for rho in self.rho]
        return self.b # optional


    def get_B0(self):
        rho = 0
        Rin, zin = sf.rho2rz(eqm = self.equ, rho_in = rho, t_in = self.moment, coord_in = 'rho_pol', all_lines = False )
        Rin,zin = Rin[0][0][0], zin[0][0][0]
        br, bz, bt = sf.rz2brzt(eqm = self.equ, r_in = Rin, z_in = zin, t_in = self.moment)
        bp = np.hypot(br, bz)
        bp, bt = bp[0][0], bt[0]
        b0 = np.hypot(bp, bt)[0]
        return b0

    def getNeFromTime(self, t: float) -> list:
        idx = np.where(self.time_ida >= t)[0][0]
        return self.ne[:,idx]

    def getTempFromTime(self, t: float) -> tuple:
        idx1 = np.where(self.time_ida >= t)[0][0]
        idx2 = np.where(self.time_idi >= t)[0][0]
        return self.te[:, idx1], self.ti[:, idx2]

    def getDataFromTime(self, t: float) -> list:
        ne = self.getNeFromTime(t)
        te, ti = self.getTempFromTime(t)
        b = self.setbFromTime(t)
        return ne,te,ti,b

    def getNeListFromTimeframe(self, t1: float, t2: float) -> list:
        idx1 = np.where(self.time_ida >= t1)[0][0]
        idx2 = np.where(self.time_ida >= t2)[0][0]
        return np.transpose(self.ne[:,idx1:idx2])

    def getAvgNeFromTimeframe(self, t1: float, t2: float) -> list:
        idx1 = np.where(self.time_ida >= t1)[0][0]
        idx2 = np.where(self.time_ida >= t2)[0][0]
        return np.average(self.ne[:,idx1:idx2], axis=1)

    def getIdxDeltaFromTimeframe(self, t1: float, t2: float) -> float:
        if t1 == t2:
            return 1
        else:
            idx1 = np.where(self.time_ida >= t1)[0][0]
            idx2 = np.where(self.time_ida >= t2)[0][0]
            return abs(idx2-idx1)

    def setData(self, moment):
        self.moment = moment
        self.idx_ida = np.where(self.time_ida >= moment)[0][0]
        self.idx_idi = np.where(self.time_idi >= moment)[0][0]
        self.idx_idg = np.where(self.time_idg >= moment)[0][0]
        self.idx_equ = np.where(self.time_equ >= moment)[0][0]
        self.idx_mai = np.where(self.time_mai >= moment)[0][0]
        self.bt = self.Bt[self.idx_mai] # We take same toroidal B for every rho
        self.b_0 = self.get_B0() # We set b for rho = 0
