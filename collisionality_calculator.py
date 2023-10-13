"""
    electron-ion collisionality-calculator
    BY JOHANNA ZACH
"""

import aug_sfutils as sf
import numpy as np
import matplotlib.pyplot as plt
import math
from data_reader import Data
from scipy.interpolate import interp1d
from scipy import interpolate

e0=8.85*10**(-12) #elektrische Feldkonstante
me=9.11*10**(-31) #Elektronenmasse
mi=3.34*10**(-27) #masse deuterium
e=1.6*10**(-19) #Elementarladung


class Collisionality:
    def __init__(self, data: Data):
        self.data = data

    # This function tries to read a shotfile independent of a specific experiment. It does that by iterating through a list of experiments and testing for testparam being available.
    def reliable_get_SF(self, shot,diagnostic,exps,testparam):
        for exp in exps:
            ret = sf.SFREAD(shot,diagnostic,exp=exp)
            if ret(testparam) is not None:
                return ret

    def elm_filt(self, data, t, shotnumber, elm_pl = 0.006, elm_mi = 0.001):
        elm = sf.SFREAD(shotnumber, 'ELM')
        t_bELM = elm('t_begELM')
        idx = np.array([])
        for T_b in t_bELM:
            idx = np.append(idx, np.where(np.logical_and(t>=T_b-elm_mi, t<=T_b+elm_pl)))
        new_dat = np.delete(data, idx.astype("int"), axis = 1)
        new_time = np.delete(t, idx.astype("int"))
        return new_dat, new_time

    def get_collisionalty(self, shot: int, t: float, ne: list = None):
        t = (t, t+0.01)

        time = self.data.time_ida
        ne = self.data.ne
        rho = self.data.rho
        Te = self.data.te
        q95 = self.data.q95
        TIMEF = self.data.timef
        Zeff = self.data.zeff
        timeZeff = self.data.timezeff
        rhop = self.data.rhop
        Ti = self.data.ti
        timeTi = self.data.time_idi
        rhopol = self.data.rhopol
        R = self.data.r
        Raus = self.data.raus
        Rin = self.data.rin
        timef = self.data.timef
        Roben = self.data.roben
        Runten = self.data.runten
        LineInfo = self.data.lineinfo
        Btf = self.data.Bt
        tmag = self.data.time_mai
        I = self.data.i
        tmag1 = self.data.tmag1

        q = self.data.q #das ist q_tor
        
        Zimp = LineInfo['Z0'][0] #Information about atomic number of main impurity

        q=q*(-1)
        q=q.T
        map=sf.mapeq
        rho_pol=map.rho2rho(ide, ide.rho_tor_n, coord_out='rho_pol')
        rho_pol=rho_pol.T
        time_ide=ide.time

        idx5=np.where(time_ide>=t[0])[0][0]
        idx6=np.where(time_ide>=t[1])[0][0]
        q1=q[:,idx5:idx6].flatten()
        rho_pol1=rho_pol[:,idx5:idx6].flatten()
        sorti=np.argsort(rho_pol1)
        q11=q1[sorti]
        rho_pol11=rho_pol1[sorti]
        idex2=np.where(rho[:,0]>=rho_pol11[-1])[0][0]
        idex22=np.where(rho[:,0]>=rho_pol11[0])[0][0]
        spl=interpolate.UnivariateSpline(rho_pol11,q11,k=3, s=1000000000, ext=3)
        xnew = rho[idex22:idex2,1]
        qfinal=spl(xnew)
        q_new=interp1d(xnew,qfinal,kind='cubic',fill_value="extrapolate") #interpolation of safety-factor
        q=q_new(rho[:,0])

        q95_new=interp1d(TIMEF,q95,kind='cubic',fill_value="extrapolate") #interpolation of safety-factor
        q95=q95_new(time)
        R_new=interp1d(TIMEF,R,kind='cubic',fill_value="extrapolate") #major plasma radius
        R=R_new(time)

        Raus_new=interp1d(TIMEF,Raus,kind='cubic',fill_value="extrapolate")
        Raus=Raus_new(time)
        Rin_new=interp1d(TIMEF,Rin,kind='cubic',fill_value="extrapolate")
        Rin=Rin_new(time)
        Roben_new=interp1d(TIMEF,Roben,kind='cubic',fill_value="extrapolate")
        Roben=Rin_new(time)
        Runten_new=interp1d(TIMEF,Runten,kind='cubic',fill_value="extrapolate")
        Runten=Rin_new(time)
        a=((Raus-Rin)/2) #minor plasma radius ###
        k=((Roben+Runten)/2)
        k1=np.power(((1+(k**2))/2),(1/2)) #=kappa

        Zeff_new=interp1d(timeZeff,Zeff[:,1],kind='cubic',fill_value="extrapolate") #interpolation of Zeff
        Zeff=Zeff_new(time)
        Ti_neu=interpolate.interp2d(timeTi, rhopol[:,0], Ti, kind='cubic') #interpolation of ion-temperature
        Ti=Ti_neu(time, rho[:,0])
        Btf_new=interp1d(tmag,Btf,kind='cubic',fill_value="extrapolate") #interpolation of Zeff
        Btf=Btf_new(time)
        I_new=interp1d(tmag1,I,kind='cubic',fill_value="extrapolate") #interpolation of Zeff
        I=I_new(time)
        qloc=2*math.pi*(a**2)*Btf*(-1)/(1.25663*(10**(-6))*R*I) #lokaler safetyfactor

        idx1=np.where(time>=t[0])[0][0] #addresses time-intervall given in line 8
        idx2=np.where(time>=t[1])[0][0]
        idx3=np.where(rho>=1)[0][0] #separatrix

        if shot==37191:
            Te_gefiltert, time_gefiltert = self.elm_filt(Te, time, shot)
            Te_gefiltert_new=interp1d(time_gefiltert,Te_gefiltert,kind='cubic',fill_value="extrapolate") #interpolation of safety-factor
            Te_gefiltert=Te_gefiltert_new(time)
            Ti_gefiltert, time_gefiltert = self.elm_filt(Ti, time, shot)
            Ti_gefiltert_new=interp1d(time_gefiltert,Ti_gefiltert,kind='cubic',fill_value="extrapolate") #interpolation of safety-factor
            Ti_gefiltert=Ti_gefiltert_new(time)
            ne_gefiltert, time_gefiltert = self.elm_filt(ne, time, shot)
            ne_gefiltert_new=interp1d(time_gefiltert,ne_gefiltert,kind='cubic',fill_value="extrapolate") #interpolation of safety-factor
            ne_gefiltert=ne_gefiltert_new(time)

        ne = np.median(ne[:,idx1:idx2], axis=1) if ne.ndim >= 2 else ne #averaging over the given time interval
        Te=np.median(Te[:,idx1:idx2], axis=1)
        q95=np.median(q95[idx1:idx2,], axis=0)
        Zeff=np.median(Zeff[idx1:idx2,], axis=0)
        R=np.median(R[idx1:idx2,], axis=0)
        a=np.median(a[idx1:idx2,], axis=0) ###
        Ti=np.median(Ti[:,idx1:idx2], axis=1)
        Btf=np.median(Btf[idx1:idx2,], axis=0)
        I=np.median(I[idx1:idx2,], axis=0)
        qloc=np.median(qloc[idx1:idx2,], axis=0)
        kappa=np.median(k1[idx1:idx2,], axis=0)

        numerat=Zimp-Zeff #calculation of n_i
        denom=Zimp-1
        quot=np.divide(numerat,denom)
        ni=np.multiply(ne,quot)

        c1=np.array(np.log(np.multiply(np.power(ne*10**(-6),1/2),np.power(Te,-1)))) #calculation of coulomb_logarithm
        c=24-c1

        aspectRatio=a/R
        l=9.2123*10**(-18) #=e(Â²)/(8*sqrt(2)*Pi*epsilon0(Â²))

        f1=l*R #calculation of collisionality
        f2=np.multiply(f1,q95)
        #f2=np.multiply(f1,q)
        f3=np.multiply(f2,ni)
        v1=np.multiply(f3,c)
        v2=np.multiply(np.power(aspectRatio,(3/2)),np.power(Te,2))
        collisionality=np.divide(v1,v2) ##q95
        ####
        ####
        ####
        #SOL calculations
        vorfaktor=(((4*math.pi*8.85*10**(-12))/((1.6*10**(-19))**2))**2) #passt
        zahler=(2*me)**(1/2)*((Te*e)**(3/2))
        nenner=2*math.pi*ni*c
        tau_ei=((vorfaktor*zahler)/nenner) ##tau ei
        cs=(((Te+Ti)*e)/(me+mi))**(1/2)

        collsol=(qloc*R*kappa*math.pi)/(cs*tau_ei) ##qloc, sonst normale sol formel ##ca Faktor 20 grÃ¶Ãer als confined
        ####
        vtherm=(((Te)*e)/(me))**(1/2)
        coll1=(qloc*R*kappa*math.pi)/(vtherm*tau_ei) ##SOL formel mit qloc, und v=sqrt(Te/me) #ca Faktor 2 grÃ¶Ãer als confined
        ####
        coll2=(qloc*R*math.pi)/(vtherm*tau_ei) ##SOL formel mit qloc, v=sqrt(Te/me) aber ohne kappa #ca wie coll1

        return collisionality
