import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np

xfilename = '/afs/ipp/home/s/slech/Documents/BSC_Calc_Lechner/Compare_Models_with_Shot38550/dunne/x_val'
yfilename = '/afs/ipp/home/s/slech/Documents/BSC_Calc_Lechner/Compare_Models_with_Shot38550/dunne/y_val'
xlist, ylist = [], []

with open(xfilename) as f:
    for line in f:
        values = line.strip().split()
        for val in values:
            xlist.append(float(val))

with open(yfilename) as f:
    for line in f:
        values = line.strip().split()
        for val in values:
            if val=='NaN' or val=='-NaN':
                val=0
            ylist.append(float(val))
xlist, ylist = np.array(xlist), np.array(ylist)
plt.plot(xlist, ylist)
plt.show()
