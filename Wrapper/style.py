import matplotlib.pyplot as plt
import matplotlib

'''
    Style choices for making plots how I like them
'''

# set tex font
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
})

# default ticks, font sizes
matplotlib.rcParams['figure.figsize'] = (7, 5)
matplotlib.rcParams["axes.labelsize"] = 18
matplotlib.rcParams["axes.titlesize"] = 18
matplotlib.rcParams["legend.fontsize"] = 16
matplotlib.rcParams["figure.titlesize"] = 30
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14

# default colour cycle
matplotlib.rcParams['axes.prop_cycle'] = plt.cycler(color=['black', 'firebrick', 'slateblue', "forestgreen", "goldenrod"])

# legend
matplotlib.rcParams['legend.frameon'] = False
matplotlib.rcParams["legend.handletextpad"] = 0.3
matplotlib.rcParams["legend.handlelength"] = 0.8
matplotlib.rcParams["legend.fontsize"] = 14