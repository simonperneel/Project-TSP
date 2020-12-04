# ---------------------------------------#
#           PYTHON PLOT                  #
# ---------------------------------------#

# use Python interpreter 3.7
# install libraries in terminal with following command:
# pip install pandas matplotlib
 
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import ScalarFormatter

# Configure figure size #
width = 20                  # inches
fontsize = 11               # pt
axis_linewidth = 0.4        # pt

golden_mean = (math.sqrt(5)-1.0)/2.0  # aesthetic ratio
height = width * golden_mean  # inches

spines = 0  # set 1 for top and right axis

# Set rc parameters #
mpl.use("pgf")
mpl.rcParams.update({
        'axes.spines.right': spines, # no right axis
        'axes.spines.top': spines, # no top axis
        'axes.axisbelow': True,
        'backend': 'ps',
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'legend.fontsize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'axes.linewidth' : axis_linewidth,
        'text.usetex': True,
        "pgf.rcfonts": False,
        "pgf.texsystem": "lualatex",
        'font.family': 'serif',
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts
            r"\usepackage[T1]{fontenc}",        # plots will be generated
            r"\usepackage[detect-all]{siunitx}" # to use si units
        ]
    })
#-----------------------------#
# Reading and processing data #
#-----------------------------#
data = pd.read_csv('r0680462.csv', delimiter=',', header=1)
# read csv into pandas dataframe
# choose right delimiter: , ;
# first line of datafile has to be column names
# this names are also used for plotting
# example Time,Current

#-----------------#
#    Plotting     #
#-----------------#
plt.figure(1, figsize=(width,height))
plt.autoscale(enable=True, axis='both', tight=None)
ax = plt.gcf().gca()

#plot with pandas
data.plot(x='# Iteration', y=' Mean value', kind='line', label='label', c='dimgray', linewidth=1, ax=ax)
data.plot(x='# Iteration', y=' Best value', kind='line', label='label', c='dimgray', linewidth=1, ax=ax)
# kind of plots: 'line', 'hist', 'scatter', 'bar'
# nice colours: 'dimgrey', 'navy', 'black', 'darkgreen', ...
# s parameter to change marker of scatterplot

#--------------------#
# Change plot layout #
#--------------------#

#title
plt.title('My graph')

#light grid
plt.grid(alpha=0.3, linewidth=0.3)

#scaling
plt.yscale('linear')  #linear, log

#range
plt.xlim()
plt.ylim()

#legend
plt.legend()

#label
plt.ylabel("ylabel []")
plt.xlabel("xlabel []")

#----------------------#
# Save and show figure #
#----------------------#
plt.savefig('xxx.pdf', format='pdf')  # formats: pdf, png,svg,...
#plt.show()
