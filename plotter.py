# -*- coding: iso-8859-1 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy import gradient as grad
from datetime import datetime
import time
from data_reader import Data

class Plot:
    """
    @ class information
        - This class can plot different data types (density, bootstrap)
        - CONSTRUCTOR: pass data set
        - MODES:
            1) ne: only plot ne and ne gradient profile
            2) bs: only plot bs and bs gradient profile
            3) stacked: plot both bs and ne and corresponding gradients
    
    @ methods to call
        - setParams() - ARGUMENTS:
            * time: needed for plot title
            * iterations: how many manipulated profile you want to plot, needed for coloring
            * ne: original ne profile
            * p1, p2: plotting boundaries, pass None if you don't want to
    
    """

    def __init__(self, data: Data, mode: str = None):
        # Set parameters
        self.neMode, self.bsMode, self.nePlotted, self.bsPlotted = False, False, False, False
        print(f'Set plotter mode!') if not self.setMode(mode) else None
        # plt.rcParams["figure.figsize"] = (10, 8) # reset with: plt.rcParams.update(plt.rcParamsDefault)
        self.rho, self.ne, self.shot = data.rho, data.ne, data.shot
        self.rho1, self.rho2 = self.rho[0], self.rho[-1]
        self.ne_mCount, self.bs_mCount, self.colli_mCount = 0, 0, 0
        # AXIS TITLES
        self.AXIS_RHO = r'$\rho_{\mathrm{p}}$'
        self.AXIS_NE = r'$\langle n_{\mathrm{e}} \rangle $ [m$^{-3}$]'
        self.AXIS_NE_GRAD = r'$\langle \nabla n_{\mathrm {e}} \rangle $ [m$^{-3}$]'
        self.AXIS_BS = r'$\langle j_\mathrm{BS}\rangle $ [MA/m$^2$]'
        self.AXIS_TEMP = r'$\langle T \rangle $ [eV]'
        self.AXIS_TEMP_GRAD = r'$\langle \nabla T \rangle $ [eV/W]'
        self.AXIS_COLLI = r'$\langle \nu* \rangle $'
        self.PLOT_TITLE = f'ASDEX Upgrade #{str(self.shot)}' # default
        # LABEL NAMES
        self.LABEL_NE = r'$\mathregular{n_e}$ original'
        self.LABEL_NE_M = r'$\mathregular{n_e}$ manipulated'
        self.LABEL_NE_GRAD = r'$\mathregular{\nabla n_e}$ original'
        self.LABEL_NE_M_GRAD = r'$\mathregular{\nabla n_e}$ manipulated'
        self.LABEL_BS = r'$\mathregular{B_s}$ ($\mathregular{n_e}$ orig.)'
        self.LABEL_BS_M = r'$\mathregular{B_s}$ ($\mathregular{n_e}$ man.)'
        self.LABEL_BS_P = r'$\mathregular{B_s}$ Peeters Method'
        self.LABEL_BS_R = r'$\mathregular{B_s}$ Redl Method'
        self.LABEL_TE =  r'$\mathregular{T_e}$'
        self.LABEL_TI = r'$\mathregular{T_i}$'
        self.LABEL_TE_GRAD = r'$\mathregular{\nabla T_e}$'
        self.LABEL_TI_GRAD = r'$\mathregular{\nabla T_i}$'
        self.LABEL_COLLI = r'$\mathregular{\nu*}$ ($\mathregular{n_e}$ orig.)'
        self.LABEL_COLLI_M = r'$\mathregular{\nu*}$ ($\mathregular{n_e}$ man.)'
        self.TITLE_FONTSIZE = 14
        self.LEGEND_FONTSIZE = 8#None
        # COLORS https://matplotlib.org/stable/gallery/color/named_colors.html
        self.COLOR_NE = 'blue'
        self.COLOR_NE_M = 'red'
        self.COLOR_NE_GRAD = self.COLOR_NE
        self.COLOR_NE_M_GRAD = self.COLOR_NE_M
        self.COLOR_BS = self.COLOR_NE
        self.COLOR_BS_M = self.COLOR_NE_M
        self.COLOR_TE = 'green'
        self.COLOR_TI = 'darkorange'
        self.COLOR_LEGENDBOX = 'aliceblue'
        # LINE STYLES
        self.LINESTYLE_NE = 'dashed'
        self.LINESTYLE_NE_M = 'solid'
        self.LINESTYLE_NE_GRAD = self.LINESTYLE_NE
        self.LINESTYLE_NE_M_GRAD = self.LINESTYLE_NE_M
        self.LINESTYLE_BS = self.LINESTYLE_NE
        self.LINESTYLE_BS_M = self.LINESTYLE_NE_M
        self.LINESTYLE_TE = self.LINESTYLE_NE
        self.LINESTYLE_TI = self.LINESTYLE_NE_M
        # LINED WIDTHS
        self.LINEWIDTH_NE = 1.5 # default
        self.LINEWIDTH_NE_M = 1
        self.LINEWIDTH_NE_GRAD = 1.5
        self.LINEWIDTH_NE_M_GRAD = 1
        self.LINEWIDTH_BS = 1.5
        self.LINEWIDTH_BS_M = 1

    def setMode(self, mode: str):
        if mode == 'ne':
            self.fig, (self.axNe, self.axNeGrad) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(5, 6))
            self.neMode = True
        elif mode == 'bs':
            self.fig, (self.axBs) = plt.subplots(nrows=1, ncols=1)
            self.bsMode = True
        elif mode == 'stacked':
            self.fig, ((self.axNe,self.axTemp), (self.axNeGrad,self.axTempGrad), (self.axBs,self.axColli)) = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(10, 9))
            self.neMode = self.bsMode = True
        else:
            raise Exception("Invalid mode")
        self.setIterations(1) #default
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        return True

    def setTime(self, time: float):
        self.PLOT_TITLE = f'ASDEX Upgrade #{str(self.shot)} at time = {time} s'

    def setPlottingBoundaries(self, rho1: float = None, rho2: float = None):
        self.rho1 = rho1 if rho1 else None
        self.rho2 = rho2 if rho2 else None

    def setIterations(self, number: int = 1):
        # defines how many maniputed profiles will be plot
        # is needed for correct colour scheming of manipulated profiles
        self.alphaList = np.flip(np.delete(np.linspace(0,1,number+1),0))

    def setParams(self, time, iterations, ne, p1, p2):
        self.setTime(time)
        self.setIterations(iterations)
        self.addNeOrig(ne)
        self.setPlottingBoundaries(p1,p2)

    def setAxisNe(self):
        # ne
        #self.axNe.set_xlabel(xlabel=self.AXIS_RHO) if not self.bsMode else None
        self.fig.suptitle(self.PLOT_TITLE, fontsize=self.TITLE_FONTSIZE)
        self.axNe.set_ylabel(ylabel=self.AXIS_NE)
        self.axNe.set_xlim([self.rho1, self.rho2])
        # grad ne
        self.axNeGrad.set_xlabel(xlabel=self.AXIS_RHO) if not self.bsMode else None
        self.axNeGrad.set_ylabel(ylabel=self.AXIS_NE_GRAD)
        self.axNeGrad.set_xlim([self.rho1, self.rho2])
        self.nePlotted = True

    def setAxisBs(self):
        self.fig.suptitle(self.PLOT_TITLE, fontsize=self.TITLE_FONTSIZE)
        self.axBs.set_xlabel(xlabel=self.AXIS_RHO)
        self.axBs.set_ylabel(ylabel=self.AXIS_BS)
        self.axBs.set_xlim([self.rho1, self.rho2])
        self.bsPlotted = True

    def setAxisColli(self):
        self.fig.suptitle(self.PLOT_TITLE, fontsize=self.TITLE_FONTSIZE)
        self.axColli.set_xlabel(xlabel=self.AXIS_RHO)
        self.axColli.set_ylabel(ylabel=self.AXIS_COLLI)
        self.axColli.set_xlim([self.rho1, self.rho2])


    def setAxisTemp(self):
        self.axTemp.set_ylabel(ylabel=self.AXIS_TEMP)
        self.axTempGrad.set_ylabel(ylabel=self.AXIS_TEMP_GRAD)
        #self.axTemp.set_yscale('log')
        #self.axTempGrad.set_yscale('log')

    def setAxisColli(self, c: list = None):
        self.axColli.set_xlabel(xlabel=self.AXIS_RHO)
        self.axColli.set_ylabel(ylabel=self.AXIS_COLLI)
        if c is not None:
            y_min = min(c[(self.rho >= self.rho1) & (self.rho <= self.rho2)])
            y_max = max(c[(self.rho >= self.rho1) & (self.rho <= self.rho2)])
            padding = 0.1
            self.axColli.set_ylim(y_min - padding, y_max + padding)

    def addNeOrig(self, ne: list):
        if not self.neMode:
            return
        self.axNe.plot(self.rho, ne, color=self.COLOR_NE, label=self.LABEL_NE, linestyle=self.LINESTYLE_NE, linewidth=self.LINEWIDTH_NE)
        self.axNeGrad.plot(self.rho, grad(ne), color=self.COLOR_NE_GRAD, label=self.LABEL_NE_GRAD, linestyle=self.LINESTYLE_NE_GRAD, linewidth=self.LINEWIDTH_NE_GRAD)
        self.setAxisNe()

    def addNeMan(self, ne_m: list, plotGrad=True):
        if not self.neMode:
            return
        tempLabel=self.LABEL_NE_M if self.ne_mCount == 0 else None
        self.axNe.plot(self.rho, ne_m, color=self.COLOR_NE_M, label=tempLabel, linestyle=self.LINESTYLE_NE_M, linewidth=self.LINEWIDTH_NE_M, alpha=self.alphaList[self.ne_mCount])
        if plotGrad:
            tempLabel=self.LABEL_NE_M_GRAD if self.ne_mCount == 0 else None
            self.axNeGrad.plot(self.rho, grad(ne_m), color=self.COLOR_NE_M_GRAD, label=tempLabel, linestyle=self.LINESTYLE_NE_M_GRAD, linewidth=self.LINEWIDTH_NE_M_GRAD, alpha=self.alphaList[self.ne_mCount])
        self.setAxisNe()
        self.ne_mCount += 1

    def addTemp(self, Temp: tuple):
        te,ti=Temp
        self.axTemp.plot(self.rho, te, color=self.COLOR_TE, label=self.LABEL_TE, linestyle=self.LINESTYLE_TE)
        self.axTemp.plot(self.rho, ti, color=self.COLOR_TI, label=self.LABEL_TI, linestyle=self.LINESTYLE_TI)
        self.axTempGrad.plot(self.rho, grad(te), color=self.COLOR_TE, label=self.LABEL_TE_GRAD, linestyle=self.LINESTYLE_TE)
        self.axTempGrad.plot(self.rho, grad(ti), color=self.COLOR_TI, label=self.LABEL_TI_GRAD, linestyle=self.LINESTYLE_TI)
        self.setAxisTemp()

    def addColliOrig(self, c: list):
        if not self.bsMode:
            return
        self.axColli.plot(self.rho, c, color=self.COLOR_BS, label=self.LABEL_COLLI, linestyle=self.LINESTYLE_BS, linewidth=self.LINEWIDTH_BS)
        self.setAxisColli(c)

    def addColliMan(self, c: list):
        if not self.bsMode:
            return
        tempLabel=self.LABEL_COLLI_M if self.colli_mCount == 0 else None
        self.axColli.plot(self.rho, c, color=self.COLOR_BS_M, label=tempLabel, linestyle=self.LINESTYLE_BS_M, linewidth=self.LINEWIDTH_BS_M, alpha=self.alphaList[self.colli_mCount])
        self.setAxisColli()
        self.colli_mCount += 1

    def addBsOrig(self, bs: list):
        if not self.bsMode:
            return
        self.axBs.plot(self.rho, bs, color=self.COLOR_BS, label=self.LABEL_BS, linestyle=self.LINESTYLE_BS, linewidth=self.LINEWIDTH_BS)
        self.setAxisBs()

    def addBasics(self, bs, c, temp):
        self.addBsOrig(bs)
        self.addColliOrig(c)
        self.addTemp(temp)

    def addBsMan(self, bs_m: list):
        if not self.bsMode:
            return
        tempLabel=self.LABEL_BS_M if self.bs_mCount == 0 else None
        self.axBs.plot(self.rho, bs_m, color=self.COLOR_BS_M, label=tempLabel, linestyle=self.LINESTYLE_BS_M, linewidth=self.LINEWIDTH_BS_M, alpha=self.alphaList[self.bs_mCount])
        self.setAxisBs()
        self.bs_mCount += 1

    def show(self):
        self.setLegend()
        plt.show()

    def save(self, filename: str = None):
        self.setLegend()
        if not filename:
            now = datetime.now()
            current_time = now.strftime("%H_%M_%S")
            ms = round(time.time() * 1000 % 12)
            filename = f'{current_time}_{ms}_{self.shot}'
        self.fig.savefig(fname=f'/home/IPP-AD/slech/Documents/BSC_Calc_Lechner/{filename}.pdf', dpi=200,format='pdf', bbox_inches='tight', transparent=True) #png

    def setLegend(self):
        if self.neMode:
            self.axNe.legend(fontsize=self.LEGEND_FONTSIZE, facecolor=self.COLOR_LEGENDBOX)
            self.axNeGrad.legend(fontsize=self.LEGEND_FONTSIZE, facecolor=self.COLOR_LEGENDBOX)
        if self.bsMode:
            self.axBs.legend(fontsize=self.LEGEND_FONTSIZE, facecolor=self.COLOR_LEGENDBOX)
        if self.neMode and self.bsMode:
            self.axTemp.legend(fontsize=self.LEGEND_FONTSIZE, facecolor=self.COLOR_LEGENDBOX)
            self.axTempGrad.legend(fontsize=self.LEGEND_FONTSIZE, facecolor=self.COLOR_LEGENDBOX)
            self.axColli.legend(fontsize=self.LEGEND_FONTSIZE, facecolor=self.COLOR_LEGENDBOX)

    def getPlot(self):
        return plt
