# -*- coding: iso-8859-1 -*-
import aug_sfutils as sf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import density_manipulator
import plotter
from numpy import sqrt
from numpy import gradient
from scipy.interpolate import interp1d as ip1d

class Shot:
    def __init__(self, shot):
        self.shot = shot
        self.exp = 'lrado'
        self.startUp()

    def startUp(self) -> bool:
        if not self.readDataFromServer():
            print(f'Error at reading Data from Server. Please try another shot.')
        self.LABEL_RHO = 'Rho / (no unit)'
        self.LABEL_DENSITY = r'Density / $\mathregular{(m^{-3})}$'
        self.LABEL_BOOTSTRAP = r'Bootstrapcurrent / (unit unclear)'

    def calcBootstrapAtSpecificTime(self, time):
        # set data
        self.setData(time)
        # set up x axis
        x_rho = np.linspace(0, self.rho[-1], len(self.rho))
        # get bs
        bs = self.calcBootstrap()
        # plot at every time
        plt.plot(x_rho, bs)
        plt.xlabel(self.LABEL_RHO)
        plt.ylabel(self.LABEL_BOOTSTRAP)
        plt.show()


    def calcBootstrapOverTimeframe(self, t1=3, t2=5, interval_length=1):
        # set up time
        # t1,t2 in sec, intervall length in ms
        interval_length /= 1000
        time_frame = np.arange(t1, t2, interval_length)
        # init bs_list depending on time: [bs[t1],...bs[t2]]
        bs = []
        # set up x axis
        x_rho = np.linspace(0, self.rho[-1], len(self.rho))
        # get bs for each time
        for time in time_frame:
            self.setData(time)
            bs.append(self.calcBootstrap())
        # plot at every time
        for bs_at_time in bs:
            plt.plot(x_rho, bs_at_time)
        plt.title("Bootstrap current")
        plt.xlabel("rho")
        plt.ylabel("bootstrap current")
        plt.show()

    def readDataFromServer(self):
        names, status = ["LRADO", "GHARR", "AUGD"], False
        for name in names:
            try:
                equ = sf.EQU(self.shot, diag="IDE", exp=name)
                status = True
                break
            except AttributeError:
                continue
        if not status:
            print(f'EQU cannot be read.')
            return False

        ida = sf.SFREAD(self.shot, 'ida', exp=self.exp)
        idi = sf.SFREAD(self.shot, 'idi', exp='AUGD')  # on idi there is only AUGD possible
        idg = sf.SFREAD(self.shot, 'idg', exp=self.exp)
        mai = sf.SFREAD("MAI", self.shot)

        self.equ = equ
        self.time_ida = ida('time')  # time for ida
        self.time_idi = idi('time')  # time for idi
        self.time_idg = idg('TIMEF') # time for idg
        self.time_mai = mai("T-MAG-1") # time for mai
        self.ne = ida('ne')  # electon density
        self.rho = ida('rhop')  # radius
        self.rho = [elem[0] for elem in self.rho] # since rho values are same over time
        self.te = ida('Te')  # electron temperature
        self.ti = idi('Ti')  # ion temperature
        self.maj_rad = idg('Rmag')  # major plasma radius
        self.Bt = mai("BTF")
        self.b = equ.get_profile("Bave")
        self.time_equ = equ.time

        return True

    def get_b(self, rho_idx):
        # this function returns b for given rho
        rho = self.rho[rho_idx]
        Rin, zin = sf.rho2rz(eqm = self.equ, rho_in = rho, t_in = self.moment, coord_in = 'rho_pol', all_lines = False )
        Rin = [elem[0] for elem in Rin]
        zin = [elem[0] for elem in zin]
        br, bz, bt = sf.rz2brzt(eqm = self.equ, r_in = Rin, z_in = zin, t_in = self.moment)
        bp = np.hypot(br, bz)
        bp = bp[0][0]
        blist = np.hypot(bp, bt)
        blist = blist[0]
        b_max, b_min = np.max(blist, initial=1), np.min(blist, initial=1)
        b = (b_max - b_min) / (b_max + b_min)
        return b

    def get_Bt(self):
        bt = self.Bt[self.idx_mai]
        return bt

    def get_B0(self):
        rho = 0
        Rin, zin = sf.rho2rz(eqm = self.equ, rho_in = rho, t_in = self.moment, coord_in = 'rho_pol', all_lines = False )
        Rin = [elem[0] for elem in Rin]
        zin = [elem[0] for elem in zin]
        br, bz, bt = sf.rz2brzt(eqm = self.equ, r_in = Rin, z_in = zin, t_in = self.moment)
        bp = np.hypot(br, bz)
        bp = bp[0][0]
        bt = bt[0]
        b0 = np.hypot(bp, bt)[0]
        return b0

    def setData(self, moment):
        self.idx_ida = np.where(self.time_ida >= moment)[0][0]
        self.idx_idi = np.where(self.time_idi >= moment)[0][0]
        self.idx_idg = np.where(self.time_idg >= moment)[0][0]
        self.idx_equ = np.where(self.time_equ >= moment)[0][0]
        self.idx_mai = np.where(self.time_mai >= moment)[0][0]
        self.moment = moment
        self.bt = self.get_Bt() # We take same toroidal B for every rho
        self.b_0 = self.get_B0() # We set b for rho = 0


    def wizard(self):
        print("Choose option:")
        print("[1] Plot density profile")
        print("[2] Plot manipulated arctan density profile")
        print("[3] Calculate bootstrap current")
        option = input("Type in number: ")
        if not option: # TODO ENTER DEFAULT MODE
            self.plotManipulatedDensityProfile()
        elif option == 1:
            self.plotDensityProfile()
        elif option == 2:
            self.plotManipulatedDensityProfile()
        elif option == 3:
            self.calcBootstrap()
        else:
            print("Option not available.")
            init()

    def plotDensityProfile(self):
        ne = self.ne
        rho = self.rho
        time = self.time_ida

        # extendedAnalysis = input('Want to plot specific time interval? (y/n): ')
        extendedAnalysis = 'n'


        if extendedAnalysis == 'y':
            # Get specific time interval
            startTime = int(input("Input Start Time [s]: "))
            startValue = np.where(time >= startTime)[0][0]
            stopTime = int(input("Input Stop Time [s]: "))
            stopValue = np.where(time >= stopTime)[0][0]
            # Get Stepwidth
            stepWidth = int(input("Input StephWidth [ms]: "))

        elif extendedAnalysis == 'n':
            # use default values
            minTime = np.min(time)
            maxTime = np.max(time)
            startValue = np.where(time == minTime)[0][0]
            stopValue = np.where(time == maxTime)[0][0]
            stepWidth = (stopValue - startValue) // 30  # plot 30 different lines

        plt.xlabel(self.LABEL_RHO)
        plt.ylabel(self.LABEL_DENSITY)
        colourRate = 0.1

        amountIntervalls = stopValue - startValue // stepWidth
        changeRate = 0.3 / amountIntervalls
        #print([round(elem*10**(-19),2) for elem in ne[130:195,5000]])

        for i in range(startValue, stopValue, stepWidth):
            plt.plot(rho, ne[:,i], color=(0, 0, 0, colourRate))  # the darker the line, the higher the time
            colourRate += changeRate
        ne = self.ne[:, startValue:stopValue]
        med = np.median(self.ne, axis=1)
        plt.plot(rho, med[:, ], color="red")
        print("On the plot you can see different density profiles. The later the time, the darker the line of its represented profile. The red line is the median (average over total time).")
        plt.show()

    def plotManipulatedDensityProfile(self):
        # this function manipulates a density profile
        # set parameters
        moment = 4 # moment where density profile is taken from
        rhoUntilLinear = 0 # 0.2
        rhoUntilArctan = 1.1
        arctanRange = 2
        mult_factor = 10
        constVerticalShift = 0 # 1*10**19

        self.setData(moment)
        timeIndex = self.idx_ida
        ne = [elem[timeIndex] for elem in self.ne]
        ne_m = np.array(ne)
        rho = self.rho
        rho_m = np.array(rho)
        plt.xlabel(self.LABEL_RHO)
        plt.ylabel(self.LABEL_DENSITY)

        index0_start = 0
        idxTillLinear = 0  # going to be set
        idxTillLinear_linearcorrection = 0
        idxTillArctan = 0
        index3_stop = len(rho)-1

        # find correct idxTillLinear
        for i,elem in enumerate(rho):
            if elem >= rhoUntilLinear:
                idxTillLinear = i
                break

        # find correct idxTillArctan
        for i,elem in enumerate(rho):
            if elem >= rhoUntilArctan:
                idxTillArctan = i
                break

        # Step 1: Overwrite ne_m with arctan
        # we will do this backwards for simplicity reasons
        pihalf = np.pi / 2
        newIndex = 0
        deltaIndexArctan = idxTillArctan - idxTillLinear
        deltaDensityArctan = ne_m[idxTillArctan] - ne_m[idxTillLinear]
        growRate_arctan = arctanRange / deltaIndexArctan
        for i in range(idxTillArctan, idxTillLinear - 1, -1):
            j = i - idxTillLinear
            newIndexFloat = i + np.abs(np.arctan(-arctanRange + j * growRate_arctan)) * mult_factor
            newIndex = int(newIndexFloat)
            if newIndexFloat < idxTillArctan:
                ne_m[newIndex] = ne_m[i] + constVerticalShift

        # Step 2: Overwrite the rest of ne_m with a linear function
        fill_values = np.linspace(ne[0], ne_m[newIndex], num=newIndex)
        print("check: ", fill_values)
        for i in range(newIndex):
            ne_m[i] = fill_values[i]

        # plot original RED
        plt.plot(rho, ne, color="red")
        # plot manipulated GREEN
        plt.plot(rho_m, ne_m, color='blue')  # draw linear line in core


        print(f"On the plot you can see the original (red) profile and a manipulated one (blue). time={moment}")
        plt.show()

    def calcBootstrap(self):
        ne = np.transpose(self.ne) # ne has time on index 2
        ne = ne[self.idx_ida]
        te = np.transpose(self.te) # te has time on index 2
        te = te[self.idx_ida]
        ti_o = self.ti[self.idx_idi] # ti has time on index 1
        maj_rad = self.maj_rad[self.idx_idg] # maj rad is time dependent

        # interpolate ti to achieve same shape as ne,te
        ti = []
        x = np.arange(len(ti_o))
        y = np.array([val for val in ti_o])
        predict = ip1d(x, y, kind='linear')
        x_new = np.linspace(0, len(ti_o)-1, len(self.rho)) # (0, 100, 200)
        ti = np.array([predict(i) for i in x_new])

        # bootstrap calc
        bs_current_byrho = lambda rho: (-sqrt(self.get_b(rho)) * maj_rad * self.bt) / (self.b_0) * (2.44*(te[rho]+ti[rho])*gradient(ne)[rho] + 0.69*ne[rho]*gradient(te)[rho] - 0.42*ne[rho]*gradient(ti)[rho])
        bs_current = []
        for i in range(len(self.rho)):
            elem = bs_current_byrho(i)
            bs_current.append(np.absolute(elem))
        return bs_current  # is a list bs[rho]

def testcase():
    pass

if __name__ == "__main__":
    shot1 = Shot(38550)
    time = 4
    #shot1.calcBootstrapAtSpecificTime(time)
    #shot1.calcBootstrapOverTimeframe(3,4,500)
    #shot1.plotManipulatedDensityProfile()
    shot1.plotDensityProfile()
    #shot1.wizard()


    # shot2 = Shot(38545) # Dunne
