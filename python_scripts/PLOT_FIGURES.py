import numpy as np
from scipy import stats
import pandas as pd
import os
import pickle
from scipy.stats import t

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import matplotlib.patches as patches


from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


from matplotlib.lines import Line2D


from matplotlib.font_manager import findfont, FontProperties
import seaborn as sns
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
DIR_fig=os.environ.get("DIRECTORY") + "/figures/"
DIR_data=os.environ.get("DIRECTORY") + "/data/"
# 
# 
# 
# 
# 
# 
# 
# 
# 
std_noise_all = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
beta_all = [1.2, 1.6, 2.0, 4.0, 6.0]
STATES = ["1_1"]
for beta_intra in np.array([2,4,6]):
    for beta_input in np.array([2,4,6]):

        state = "{}_{}".format(int(beta_intra),int(beta_input))
        STATES.append(state)
STATES_ORG = [['6_2', '4_2', '6_4'],['4_4', '2_2', '6_6'],['4_6', '2_4','2_6']]
STATES_NEW = ["1_1",'6_2', '4_2', '6_4','4_4', '2_2', '6_6','4_6', '2_4','2_6']
# 
# 
# 
# 
with open(DIR_data+"data_curation.pickle", 'rb') as handle:
    data = pickle.load(handle)
# 
# 
# 
# 
INPUTS = np.arange(1,10,2)/100
# 
# 
# 
# 
colors = {"1_1":"r",
 "2_2":"cyan","2_4": "lawngreen","2_6": "gold",
 "4_2": "dodgerblue", "4_4": "limegreen", "4_6": "darkgoldenrod", 
 "6_2": "blue", "6_4": "darkgreen","6_6":"saddlebrown" }




upscaling_title = ["Local-Selective ("+ r"$\bf{LS}$" + ")",
                   "Homogeneous ("+ r"$\bf{H}$" + ")",
                   "Distance-Selective ("+ r"$\bf{DS}$" + ")"]
# 
# 
# 
# 
n_columns = [1,2]
columns = ["pert","unpert"]


bintras = [2,4,6] 
binputs = [2,4,6]
# 
# 
# 
# 
markersize = 2
capsize = 2.5
capthick = 1
markeredgewidth = 1
lw = 1
# 
# 
# 
# 
params = {'legend.fontsize': 7,
         'axes.labelsize': 7,
         'axes.titlesize':7,
         'xtick.labelsize':7,
         'ytick.labelsize':7,
         'axes.linewidth': 0.4,
         'xtick.major.width':0.4,
         'ytick.major.width':0.4,
         'xtick.major.size':3,
         'ytick.major.size':3}
plt.rcParams.update(params)


plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
# 
# 
# 
# 
font = findfont(FontProperties(family=['sans-serif']))
# 
# 
# 
# 
# 
# 
# 
# 
def figure_1(left,right,top,bottom):
    fix_x = 7.08#unit in inch
    ws_x = 0.34#of the axis x lenght
    ws_y = 0.7
    distx = 0.48 #of the axis x lenght
    disty = 0.2
    ax_x = fix_x/(3+2*ws_x)
    ax_y = 0.7 * ax_x
    
    fix_y = (3.2 + 2*disty - 0.25*disty) * ax_y
    
    
    
    
    
    gs1 = GridSpec(1,1, bottom=bottom+(2.+2*disty)*ax_y/fix_y, top=-top+1, left=left+0., right=-right+(1)*ax_x/fix_x, wspace=ws_x,hspace = ws_y)
    gs2 = GridSpec(1,1, bottom=bottom+(2.+2.9*disty)*ax_y/fix_y, top=-top+1, left=left+(1.+.9*ws_x)*ax_x/fix_x, right=-right+(2.+1.2*ws_x)*ax_x/fix_x,wspace=ws_x,hspace = ws_y)
    gs22 = GridSpec(1,1, bottom=bottom+(2.+2.6*disty)*ax_y/fix_y, top=-top+1-0.05, left=left+(2.+2.*ws_x)*ax_x/fix_x, right=-right+1,wspace=ws_x,hspace = ws_y)
    gs3 = GridSpec(1,3, bottom=bottom+(1+disty)*ax_y/fix_y, top=-top+(2.+disty)*ax_y/fix_y, left=left+0, right=-right+1,wspace=ws_x,hspace = ws_y)
    gs4 = GridSpec(1,3, bottom=bottom, top=-top+(1)*ax_y/fix_y, left=left+0, right=-right+1,wspace=ws_x,hspace = ws_y)
    
    fig = plt.figure(figsize=(fix_x, fix_y))
    
    
    axA = []
    axA.append(fig.add_subplot(gs1[0]))
    axB = []
    axB.append(fig.add_subplot(gs2[0]))
    axB2 = []
    axB2.append(fig.add_subplot(gs22[0]))
    axC = []
    axD = []
    for i in range(3):
        axC.append(fig.add_subplot(gs3[i]))
        axD.append(fig.add_subplot(gs4[i]))
    
    
    
    
    
    
    
    
    
    
    
    
    axB[0].plot(1,0.8,"o",color=colors["1_1"],alpha=0.8,markersize=markersize)
    for i in [2,4,6]:
        state = "{}_2".format(i)
        axB[0].plot(i,0.8,"o",color=colors[state],alpha=0.8,markersize=markersize)
    
    
    axB[0].set_xlabel("Synaptic Upscaling of\nIntra-Excitatory Connections ("+r"$\beta_{\mathrm{intra}}$"+")")
    axB[0].set_xlim(-0,7)
    axB[0].set_xticks([1,2,4,6])
    axB[0].set_xticklabels([1,2,4,6])
    axB[0].set_ylim(0.5,2)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    for beta in [1,1.2,1.6,2,4.,6.]:
        if beta not in [1,2]:
            rel_pow = data["prestimulus_beta_analysis"]['beta:{}'.format(beta)]["rel_pow"]
            rel_pow_std = data["prestimulus_beta_analysis"]['beta:{}'.format(beta)]["rel_pow_std"]
            
        else:
            rel_pow = data["prestimulus_spontaneous_analysis"]["{}_{}".format(beta,beta)]["1_column"]["rel_pow"]
            rel_pow_std = data["prestimulus_spontaneous_analysis"]["{}_{}".format(beta,beta)]["1_column"]["rel_pow_std"]

        if beta in [1.2,1.6]:
            axB2[0].errorbar(beta,rel_pow[0],yerr=rel_pow_std[0],color="gray", fmt="o", markersize=markersize,zorder=1, capsize=2, elinewidth=0.8,capthick= 0.8)
        elif beta in [2,4.,6.]:
            state = "{}_2".format(int(beta))
            axB2[0].errorbar(beta,rel_pow[0],yerr=rel_pow_std[0],color=colors[state], fmt="o", markersize=markersize,zorder=1, capsize=2, elinewidth=0.8,capthick= 0.8)
        else:
            state = "1_1"
            axB2[0].errorbar(beta,rel_pow[0],yerr=rel_pow_std[0],color=colors[state], fmt="o", markersize=markersize,zorder=1, capsize=2, elinewidth=0.8,capthick= 0.8)
        
        
    def Fit(x):
        return 0.19 + 3.2*np.exp(-x/0.428)
        
    x = np.linspace(1,6,100)
    axB2[0].plot(x,Fit(x),"k--",zorder=1,lw=1)
        
    
    axB2[0].set_xlabel(r"$\beta_{\mathrm{intra}}$")
    axB2[0].set_ylabel("Power of\nSlow Oscillation")
    axB2[0].set_xticks([1,2,4,6])
    axB2[0].set_xticklabels([1,2,4,6])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ax_com = [axC,axD]
    
    
    
    
    
    bimodal_limit = -0.014, 0.265
    power_limit = 0,0.715
    
    for i,state in enumerate(["1_1","2_2"]):
        ax_com[i][0].plot(data["prestimulus_spontaneous_analysis"][state]["1_column"]["instance"][::10],color=colors[state],alpha=0.8)
    
        ax_com[i][0].set_xlabel("Time (s)")
        ax_com[i][0].set_xticks([0,2000,4000])
        ax_com[i][0].set_xticklabels([0,2,4])
        ax_com[i][0].set_ylim(0,30)
        ax_com[i][0].set_ylabel("Firing Rate (Hz)")
    
    
        ylimmin,ylimmax=np.inf,-np.inf
        
        hist = data["prestimulus_spontaneous_analysis"][state]["1_column"]["hist"]/100
        hist_std = data["prestimulus_spontaneous_analysis"][state]["1_column"]["hist_std"]/100
        hist_bin = data["prestimulus_spontaneous_analysis"][state]["1_column"]["hist_bin"]
    
        ax_com[i][1].plot(hist_bin,hist,color=colors[state],alpha=0.8)
        ax_com[i][1].fill_between(hist_bin,hist-hist_std,hist+hist_std,color=colors[state],alpha=0.2)
    
        ylim = ax_com[i][1].get_ylim()
        ylimmin = min(ylimmin,ylim[0])
        ylimmax = max(ylimmax,ylim[1])
        ax_com[i][1].set_xlabel("Firing Rate (Hz)")
    
    # bimodal_limit = ylimmin,ylimmax
    for i in range(2):
        ax_com[i][1].set_ylim(bimodal_limit)
    
    
    axinylimmin,axinylimmax=np.inf,-np.inf
    ylimmin,ylimmax=np.inf,-np.inf
    for i,state in enumerate(["1_1","2_2"]):
        rel_pow = data["prestimulus_spontaneous_analysis"][state]["1_column"]["rel_pow"]
        rel_pow_std = data["prestimulus_spontaneous_analysis"][state]["1_column"]["rel_pow_std"]
        ax_com[i][2].bar(np.arange(len(rel_pow)), rel_pow, yerr=rel_pow_std, align='center', alpha=0.8,color=colors[state], ecolor='black', capsize=2, error_kw={"elinewidth":0.8,"capthick": 0.8})
        ax_com[i][2].set_xticks(np.arange(len(rel_pow)))
        ax_com[i][2].set_xticklabels(["SO","Delta","Theta","Alpha","Beta","Gamma"],rotation = 45)
    
        ylim = ax_com[i][2].get_ylim()
        ylimmin = min(ylimmin,ylim[0])
        ylimmax = max(ylimmax,ylim[1])
    
    
        axinylimmin = min(axinylimmin,rel_pow[-1]-rel_pow_std[-1])
        axinylimmax = max(axinylimmax,rel_pow[-1]+rel_pow_std[-1])
    
    # power_limit = 0,0.7
    power_limit = ylimmin,ylimmax
    for i in range(2):
        ax_com[i][2].set_ylim(power_limit) 
    
        ax_com[i][1].set_ylabel("Occurrence")
    
        ax_com[i][2].set_ylabel("Power")
    
    axC[0].set_title("One Representative Trial")
    axC[1].set_title("Histogram")
    axC[2].set_title("Power Spectrum")
    
    
    
    
    
    
    axA[0].text(-0.2,2.78,r"$\bf{a}$",horizontalalignment='left', transform=axC[0].transAxes, fontsize=7)
    
    
    axA[0].axis('off')
    
    axB[0].spines[['right', 'top',"left"]].set_visible(False)
    axB[0].axes.get_yaxis().set_visible(False)
    
    lines = [Line2D([0], [0], color="white", marker='o',markersize=0.005, markerfacecolor="white") for c in range(3)]
    labels  = ["Intra-Excitatory\nConnections",
              "Inhibitory\nConnections","Noise"]
    
    # axA[0].legend(lines, labels,handlelength=3,loc='center right', bbox_to_anchor=(1.1, 0.5),fontsize=7)
    
    axB[0].text(-0.,2.78,r"$\bf{b}$",horizontalalignment='left', transform=axC[1].transAxes, fontsize=7)
    
    axB2[0].text(-0.14,2.78,r"$\bf{e}$",horizontalalignment='left', transform=axC[2].transAxes, fontsize=7)
    
    
    
    
    axC[0].text(-0.2,1.15,r"$\bf{C}$"+"(i)",horizontalalignment='left', transform=axC[0].transAxes, fontsize=7)
    axC[1].text(-0.23,1.15,"(ii)",horizontalalignment='left', transform=axC[1].transAxes, fontsize=7)
    axC[2].text(-0.23,1.15,"(ii)",horizontalalignment='left', transform=axC[2].transAxes, fontsize=7)
    axD[0].text(-0.2,1.15,r"$\bf{d}$"+"(i)",horizontalalignment='left', transform=axD[0].transAxes, fontsize=7)
    axD[1].text(-0.23,1.15,"(ii)",horizontalalignment='left', transform=axD[1].transAxes, fontsize=7)
    axD[2].text(-0.23,1.15,"(ii)",horizontalalignment='left', transform=axD[2].transAxes, fontsize=7)
    
    
    axB[0].text(0.07, 0.47, "NREM", transform=axB[0].transAxes,fontsize=7)
    axB[0].text(0.46, 0.47, "Wake (W)", transform=axB[0].transAxes,fontsize=7)


    arrow = patches.FancyArrowPatch((4, 0.95), (4, 1.22), color='black', arrowstyle='->, widthB=.25, lengthB=0.', mutation_scale=10, linewidth=0.7)
    axB[0].add_patch(arrow)
    arrow = patches.FancyArrowPatch((1.9, 0.99), (6.1, 0.99), color='black', arrowstyle='-', mutation_scale=10, linewidth=0.7)
    axB[0].add_patch(arrow)
    arrow = patches.FancyArrowPatch((2, 1.035), (2, 0.82), color='black', arrowstyle='-', mutation_scale=10, linewidth=0.7)
    axB[0].add_patch(arrow)
    axB[0].plot((6., 6.8), (0.99, 0.99), linestyle=':',color="k", linewidth=0.7)

    arrow = patches.FancyArrowPatch((1, 0.82), (1, 1.22), color='black', arrowstyle='->, widthB=.25, lengthB=0.', mutation_scale=10, linewidth=0.7)
    axB[0].add_patch(arrow)
    
    fig.savefig(DIR_fig+"fig1.svg")
# 
# 
# 
# 
left=0.052
right=0.002
top=0.
bottom=0.08


figure_1(left,right,top,bottom)
# 
# 
# 
# 
def figure_2(left,right,top,bottom):
    fix_x = 7.08 #unit in inch
    ws_x = 0.14 #of the axis x lenght
    ws_y = 0.6
    distx = 0.2 #of the axis x lenght
    disty = 0.08
    ax_x = fix_x/(3+2*ws_x)
    ax_y = 0.9 * ax_x
    
    fix_y = (2.2 + 1*disty -0.4*ws_y) * ax_y
    
    gs1 = GridSpec(1,1, bottom=bottom+(1.+1*disty)*ax_y/fix_y, top=-top+1, left=left+0., right=-right+(1+.0*ws_x)*ax_x/fix_x, wspace=ws_x,hspace = ws_y)
    gs2 = GridSpec(1,1, bottom=bottom+(1.+1.1*disty+0.*ws_y)*ax_y/fix_y, top=-top+1, left=left+(1.05+2.5*ws_x)*ax_x/fix_x, right=-right+(2+1.*ws_x)*ax_x/fix_x,wspace=ws_x,hspace = ws_y)
    gs22 = GridSpec(1,2, bottom=bottom+(1.+1.1*disty+0.*ws_y)*ax_y/fix_y, top=-top+1, left=left+(1.85+3*ws_x)*ax_x/fix_x, right=-right+1,wspace=3*ws_x,hspace = ws_y)
    gs3 = GridSpec(1,2, bottom=bottom+0, top=-top+(1.+0*disty)*ax_y/fix_y, left=left+0, right=-right+(2.+1.*ws_x)*ax_x/fix_x,wspace=ws_x,hspace = ws_y)
    gs32 = GridSpec(1,1, bottom=bottom+0, top=-top+(1.+0*disty)*ax_y/fix_y, left=left+(1.85+3*ws_x)*ax_x/fix_x, right=-right+1,wspace=ws_x,hspace = ws_y)
    
    fig = plt.figure(figsize=(fix_x, fix_y))
    
    
    axA = []
    axA.append(fig.add_subplot(gs1[0]))
    axB = []
    axB.append(fig.add_subplot(gs2[0]))
    axB2 = []
    for i in range(2):
        axB2.append(fig.add_subplot(gs22[i]))
    axC = []
    for i in range(2):
        axC.append(fig.add_subplot(gs3[i]))
    axC = np.array(axC).reshape(2,1)
    
    axC2 = []
    axC2.append(fig.add_subplot(gs32[0]))
    
    
    
    
    
    
    
    
    
    
    trianglex = [ 3,6.5, 6.5, 3 ]
    trianglex = np.array(trianglex)
    triangley = [ 1.7,1.7, 5.2, 1.7 ]
    triangley = np.array(triangley)
    
    triangle2x = [ 1.7,1.7, 5.2, 1.7 ] 
    triangle2x = np.array(triangle2x)
    
    triangle2y = [ 3.,6.5, 6.5, 3.]
    triangle2y = np.array(triangle2y)
    
    triangle3x = [ 1.7,1.7, 2.5,6.5,6.5, 5.7, 1.7 ] 
    triangle3x = np.array(triangle3x)
    triangle3y = [ 2.5,1.7, 1.7, 5.7,6.5,6.5,2.5]   
    triangle3y = np.array(triangle3y)
    
    
    
    axB[0].fill(trianglex, triangley,color="gray",alpha=0.3,lw=0)
    axB[0].fill(triangle2x, triangle2y,color="gray",alpha=0.3,lw=0)
    axB[0].fill(triangle3x, triangle3y,color="gray",alpha=0.3,lw=0)
    
    
    for state in STATES:
        bintra,binput = state.split("_")
        state = "{}_{}".format(bintra,binput)
        axB[0].plot(int(bintra),int(binput),"o",color=colors[state],alpha=0.8,markersize=markersize)
    axB[0].set_xlabel(r"$\beta_{\mathrm{intra}}$")
    axB[0].set_ylabel("Synaptic Upscaling\nof Inter-Excitatory\nConnections ("+r"$\beta_{\mathrm{inter}}$"+")")
    axB[0].set_xticks([1,2,4,6])
    axB[0].set_xticklabels([1,2,4,6])
    axB[0].set_yticks([1,2,4,6])
    axB[0].set_yticklabels([1,2,4,6])
    
    
    
    
    
    
    
    axB[0].text(0.37,.75,r"$\bf{DS}$",horizontalalignment='left', transform=axB[0].transAxes,fontsize=7)
    axB[0].text(0.75,.75,r"$\bf{H}$",horizontalalignment='left', transform=axB[0].transAxes,fontsize=7)
    axB[0].text(0.75,.4,r"$\bf{LS}$",horizontalalignment='left', transform=axB[0].transAxes,fontsize=7)
    
    
    
    
    
    
    
    
    
    
    
    
    column = "pert"
    pop = "pyr"
    
    
    state = "1_1"
    y = data['deterministic_response'][state]["1_column"]["input:0.05"]["{}".format(column)]["{}".format(pop)]["firing"]
    axB2[0].plot(y[0,::10],color=colors[state],alpha=0.8,markersize=markersize, zorder=2)
    axB2[0].set_ylim(3,28)
    
    binput = 2
    for i,bintra in enumerate(bintras):
        state = "{}_{}".format(bintra,binput)
        y = data['deterministic_response'][state]["1_column"]["input:0.05"]["pert"]["pyr"]["firing"]
        axB2[1].plot(y[0,::10],color=colors[state],alpha=0.8,markersize=markersize, zorder=2)
        
    
    
    for i in range(2):
        ylimmin,ylimmax = np.inf,-np.inf
        axB2[i].set_xlim(850,2000)
        axB2[i].set_xticks([1000,2000])
        axB2[i].set_xticklabels([0,1])
    
        ylim = axB2[i].get_ylim()
    
        ylimmin = min(ylimmin,ylim[0])
        ylimmax = max(ylimmax,ylim[1])
        
        axB2[i].fill_between([1000,1100],[4.8]*2,[28.4]*2,color="gray",alpha=0.3, linewidth=0, zorder=1)
        # axB2[i].set_ylim(ylimmin,ylimmax)
        axB2[i].set_xlabel("Time (s)")
        
    # axins = zoomed_inset_axes(axB2[1], 4, loc=5)
    # x1, x2, y1, y2 = 1000-50, 1200 ,22.5, 27.5
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    
    # for i,bintra in enumerate(bintras):
    #     state = "{}_{}".format(bintra,binput)
    #     y = data['deterministic_response'][state]["1_column"]["input:0.05"]["pert"]["pyr"]["firing"]
    #     axins.plot(y[0,::10],color=colors[state],alpha=0.8,markersize=markersize)
    #     axins.fill_between([1000,1100],[ylimmin]*2,[ylimmax+0.04*(ylimmax-ylimmin)]*2,color="gray",alpha=0.3)
    # axins.set_xticks([])
    # axins.set_yticks([])
    # mark_inset(axB2[1], axins, loc1=1, loc2=3, fc="none", ec="0.1")
    axB2[0].set_ylim(4.8,27.8)
    axB2[1].set_ylim(22,27.6)
    
    axB2[0].set_ylim(4.8,28.4)
    axB2[1].set_ylim(22,28.2)
    axB2[1].set_yticks([23,25,27])
    axB2[0].set_yticks([5,10,15,20,25])
    axB2[0].set_ylabel("Firing Rate (Hz)")
    
    axB2[0].set_xlim(900,2100)
    axB2[1].set_xlim(900,2100)
    
    axB2[0].text(0.02, 0.92, "Stimulus=50 Hz", transform=axB2[0].transAxes,fontsize=7)
    axB2[1].text(0.02, 0.92, "Stimulus=50 Hz", transform=axB2[1].transAxes,fontsize=7)
    
    
    
    
    
    
    
    
    
    
    
    
    
    data_arrow_pull = np.zeros(3,dtype=float)
    data_arrow_drive = np.zeros(3,dtype=float)
    
    
    state = "1_1"
    label = "NREM"
    y = np.zeros(len(INPUTS),dtype=float)
    for i,inpu in enumerate(INPUTS):
        y[i] = data['deterministic_response'][state]["1_column"]["input:{}".format(inpu)]["pert"]["pyr"]["firing"][0,11000]
    y = y - data['deterministic_response'][state]["1_column"]["input:{}".format(inpu)]["pert"]["pyr"]["firing"][0,9000]
    axC[0,0].plot(INPUTS,y, 'o',color=colors[state],alpha=0.8,markersize=markersize)
    axC[1,0].plot(INPUTS,y, 'o',color=colors[state],alpha=0.8,markersize=markersize)
    line1 = Line2D([], [], color="white", marker='o', markerfacecolor=colors["1_1"],alpha=0.8,markersize=markersize)
    
    line2 = []
    
    bintra = 2
    for i,binput in enumerate(binputs):
        state = "{}_{}".format(bintra,binput)
        label = "W"
        y = np.zeros(len(INPUTS),dtype=float)
        for ii,inpu in enumerate(INPUTS):
            y[ii] = data['deterministic_response'][state]["1_column"]["input:{}".format(inpu)]["pert"]["pyr"]["firing"][0,11000]
        y = y - data['deterministic_response'][state]["1_column"]["input:{}".format(inpu)]["pert"]["pyr"]["firing"][0,9000]
        axC[1,0].plot(INPUTS,y, 'o',color=colors[state],alpha=0.8,markersize=markersize)
    
        data_arrow_drive[i] = y[2]
    
    
    line2 = []
    
    binput = 2
    for i,bintra in enumerate(bintras):
        state = "{}_{}".format(bintra,binput)
        label = "W"
        y = np.zeros(len(INPUTS),dtype=float)
        for ii,inpu in enumerate(INPUTS):
            y[ii] = data['deterministic_response'][state]["1_column"]["input:{}".format(inpu)]["pert"]["pyr"]["firing"][0,11000]
        y = y - data['deterministic_response'][state]["1_column"]["input:{}".format(inpu)]["pert"]["pyr"]["firing"][0,9000]
        axC[0,0].plot(INPUTS,y, 'o',color=colors[state],alpha=0.8,markersize=markersize,label=label)
    
    
        data_arrow_pull[i] = y[2]
    
    
    for i in range(2):
        axC[i,0].set_xticks([0.01,0.05,0.09])
        axC[i,0].set_xticklabels([10,50,90])
        axC[i,0].set_yticks([0,5,10])
        if i!=0:
            axC[i,0].set_yticklabels([])
        
        axC[i,0].set_xlabel("Stimulus Intensity (Hz)")
        axC[i,0].set_ylim([0,11])
    axC[0,0].set_ylabel("Amplitude (Hz)")    
    
    
    
    # axC[0,0].plot([0.045,0.055,0.055,0.045,0.045],[0.9,0.9,5,5,0.9],"--k")
    # axC[1,0].plot([0.045,0.055,0.055,0.045,0.045],[3.5,3.5,7,7,3.5],"--k")
    
    
    inpu = 0.05
    state = "2_2"
    y1 = data['deterministic_response'][state]["1_column"]["input:{}".format(inpu)]["pert"]["pyr"]["firing"][0,11000]
    y1 = y1 - data['deterministic_response'][state]["1_column"]["input:{}".format(inpu)]["pert"]["pyr"]["firing"][0,9000]
    y1 = 0.97*y1
    state = "6_2"
    y2 = data['deterministic_response'][state]["1_column"]["input:{}".format(inpu)]["pert"]["pyr"]["firing"][0,11000]
    y2 = y2 - data['deterministic_response'][state]["1_column"]["input:{}".format(inpu)]["pert"]["pyr"]["firing"][0,9000]
    y2 = 1.2*y2
    
    axC[0,0].arrow(0.055,y1,0,y2-y1,width=0.00001, head_width=0.002, head_length=0.4, fc='k', ec='k')
    axC[0,0].text(0.6, 0.16, "pulling", transform=axC[0,0].transAxes,fontsize=7,rotation=90)
    
    inpu = 0.05
    state = "2_2"
    y1 = data['deterministic_response'][state]["1_column"]["input:{}".format(inpu)]["pert"]["pyr"]["firing"][0,11000]
    y1 = y1 - data['deterministic_response'][state]["1_column"]["input:{}".format(inpu)]["pert"]["pyr"]["firing"][0,9000]
    y1 = 1.*y1
    state = "2_6"
    y2 = data['deterministic_response'][state]["1_column"]["input:{}".format(inpu)]["pert"]["pyr"]["firing"][0,11000]
    y2 = y2 - data['deterministic_response'][state]["1_column"]["input:{}".format(inpu)]["pert"]["pyr"]["firing"][0,9000]
    y2 = 0.93*y2
    axC[1,0].arrow(0.055,y1,0,y2-y1,width=0.00001, head_width=0.002, head_length=0.4, fc='k', ec='k')
    axC[1,0].text(0.6, 0.36, "driving", transform=axC[1,0].transAxes,fontsize=7,rotation=90)
    
    
    # axC[1,0].set_title("Driving Effect")
    # axC[0,0].set_title("Pulling Effect")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    inpu = 0.05
    
    state = "{}_{}".format(1,1)
    
    y = data['deterministic_response'][state]["1_column"]["input:{}".format(inpu)]["pert"]["pyr"]["firing"][0,11000]
    y = y - data['deterministic_response'][state]["1_column"]["input:{}".format(inpu)]["pert"]["pyr"]["firing"][0,9000]
    
    
    axC2[0].plot(0,y,"o",color=colors[state],alpha=0.8,markersize=markersize)
    for k in range(len(STATES_ORG)):
        for i,state in enumerate(STATES_ORG[k]):
            y = data['deterministic_response'][state]["1_column"]["input:{}".format(inpu)]["pert"]["pyr"]["firing"][0,11000]
            y = y - data['deterministic_response'][state]["1_column"]["input:{}".format(inpu)]["pert"]["pyr"]["firing"][0,9000]
            axC2[0].plot(1+k+0.15 * (i-1),y, "o",color=colors[state],alpha=0.8,markersize=markersize)
    axC2[0].set_ylabel("Amplitude (Hz)")
    axC2[0].set_xticks([0,1,2,3])
    axC2[0].set_xticklabels(["NREM", "W ("+r"$\bf{LS}$"+")", "W ("+r"$\bf{H}$"+")", "W ("+r"$\bf{DS}$"+")"],rotation=30)
    # axC2[0].set_xticklabels(['\n N',
    #                          r"$\beta_{\mathrm{intra}}>\beta_{\mathrm{inter}}>1$", 
    #                          r"$\beta_{\mathrm{intra}}=\beta_{\mathrm{inter}}>1$", 
    #                          r"$\beta_{\mathrm{inter}}>\beta_{\mathrm{intra}}>1$"], rotation=30)
    # axC2[0].set_title("Information Differentiation")
    
    
    axC2[0].text(0.24, 0.87, "Stimulus=50 Hz", transform=axC2[0].transAxes,fontsize=7)


    ylimmin,ylimmax = np.inf,-np.inf
    for i in range(2):
        ylim = axC[i,0].get_ylim()
        ylimmin = min(ylimmin,ylim[0])
        ylimmax = max(ylimmax,ylim[1])
    axC2[0].set_ylim(ylimmin,ylimmax)
    axC2[0].set_yticks([0,5,10])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    axA[0].axis('off')
    
    
    
    
    
    axA[0].text(-0.21,1.15,r"$\bf{a}$",horizontalalignment='left', transform=axA[0].transAxes, fontsize=7)
    
    axB[0].text(-0.51,1.15,r"$\bf{b}$",horizontalalignment='left', transform=axB[0].transAxes, fontsize=7)
    
    axB2[0].text(-0.5,1.15,r"$\bf{c}$"+"(i)",horizontalalignment='left', transform=axB2[0].transAxes, fontsize=7)
    axB2[1].text(-0.28,1.15,"(ii)",horizontalalignment='left', transform=axB2[1].transAxes, fontsize=7)
    
    axC[0,0].text(-0.185,1.15,r"$\bf{d}$"+"(i)",horizontalalignment='left', transform=axC[0,0].transAxes, fontsize=7)
    axC[1,0].text(-0.08,1.15,"(ii)",horizontalalignment='left', transform=axC[1,0].transAxes, fontsize=7)
    
    axC2[0].text(-0.17,1.15,r"$\bf{e}$",horizontalalignment='left', transform=axC2[0].transAxes, fontsize=7)
    
    
    
    
    axB[0].text(0.4, 0.08, "NREM", transform=axB[0].transAxes,fontsize=7)
    axB[0].text(0.03, 0.545, "W", transform=axB[0].transAxes,fontsize=7,rotation=90)
    axB[0].set_xlim(0.3,6.8)
    axB[0].set_ylim(0.3,6.8)
    


    arrow = patches.FancyArrowPatch((1.8, 4), (0.8, 4), color='black', arrowstyle='->, widthB=.25, lengthB=0.', mutation_scale=10, linewidth=0.7)
    axB[0].add_patch(arrow)
    # arrow = patches.FancyArrowPatch((1.9, 0.99), (6.1, 0.99), color='black', arrowstyle='-', mutation_scale=10, linewidth=0.7)
    # axB[0].add_patch(arrow)
    # arrow = patches.FancyArrowPatch((2, 1.035), (2, 0.82), color='black', arrowstyle='-', mutation_scale=10, linewidth=0.7)
    # axB[0].add_patch(arrow)
    axB[0].plot((1.65, 6.), (1.65, 1.65), linestyle='-',color="k", linewidth=0.7)
    axB[0].plot((6, 6.7), (1.65, 1.65), linestyle=':',color="k", linewidth=0.7)
    axB[0].plot((1.65, 1.65),(1.65, 6.), linestyle='-',color="k", linewidth=0.7)
    axB[0].plot((1.65, 1.65),(6, 6.7), linestyle=':',color="k", linewidth=0.7)

    arrow = patches.FancyArrowPatch((1.2, 1), (2.9, 1), color='black', arrowstyle='->, widthB=.25, lengthB=0.', mutation_scale=10, linewidth=0.7)
    axB[0].add_patch(arrow)
    
    
    fig.savefig(DIR_fig+"fig2.svg")
# 
# 
# 
# 
left=0.052
right=0.008
top=0.07
bottom=0.1


figure_2(left,right,top,bottom)
# 
# 
# 
# 
def figure_3(left,right,top,bottom):
    fix_x = 7.08 #unit in inch
    ws_x = 0.06 #of the axis x lenght
    ws_y = 0.7
    distx = 0.48 #of the axis x lenght
    disty = 0.7
    ax_x = fix_x/(3+2*ws_x)
    ax_y = 0.7 * ax_x
    
    fix_y = (1 + 0*disty) * ax_y
    
    
    gs1 = GridSpec(1,1, bottom=bottom+0, top=-top+1, left=left+0, right=-right+(1)*ax_x/fix_x,wspace=ws_x,hspace = ws_y)
    gs12 = GridSpec(1,1, bottom=bottom+0, top=-top+1, left=left+(1+1.8*ws_x)*ax_x/fix_x, right=-right+(2+1.*ws_x)*ax_x/fix_x,wspace=ws_x,hspace = ws_y)
    gs2 = GridSpec(1,1, bottom=bottom+0, top=-top+1, left=left+(2+2.8*ws_x)*ax_x/fix_x, right=-right+1,wspace=ws_x,hspace = ws_y)
    
    fig = plt.figure(figsize=(fix_x, fix_y))
    
    
    axA = []
    axA.append(fig.add_subplot(gs1[0]))
    axA2 = []
    axA2.append(fig.add_subplot(gs12[0]))
    axB = []
    axB.append(fig.add_subplot(gs2[0]))
    
    
    
    
    
    
    
    
    
    
    
    
    
    n_column = 1
    column = "pert"
    pop = "pyr"
    method = "k_means"
    
    
    
    
    t_crit = -stats.t.ppf(0.05/2,10)
    
    
    
    
    
    bintra,binter = 1,1
    state = "{}_{}".format(bintra,binter)
    y = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][0]
    yerr = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][1]
    yerr = t_crit * yerr / np.sqrt(10)
    axA[0].errorbar(INPUTS,y,yerr=yerr, fmt='o',color=colors[state],capsize=capsize, markersize=markersize, capthick=capthick, lw=lw, markeredgewidth=markeredgewidth,alpha=0.8)
    
    line1 = Line2D([], [], color="white", marker='o', markerfacecolor=colors["1_1"],alpha=0.8)
    
    
    
    
    line2 = []
    
    binput = 2
    for i,bintra in enumerate([2,6]):
        state = "{}_{}".format(bintra,binput)
        y = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][0]
        yerr = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][1]
        yerr = t_crit * yerr / np.sqrt(10)
        axA[0].errorbar(INPUTS,y,yerr=yerr, fmt='o',color=colors[state],capsize=capsize, markersize=markersize, capthick=capthick, lw=lw, markeredgewidth=markeredgewidth,alpha=0.8)
    
    bintra = 2
    for i,binput in enumerate([6]):
        state = "{}_{}".format(bintra,binput)
        y = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][0]
        yerr = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][1]
        yerr = t_crit * yerr / np.sqrt(10)
        axA[0].errorbar(INPUTS,y,yerr=yerr, fmt='o',color=colors[state],capsize=capsize, markersize=markersize, capthick=capthick, lw=lw, markeredgewidth=markeredgewidth,alpha=0.8)
    
    # l = axA[0].legend([line1], ["NREM"],loc=2)
    
    
    axA[0].set_xticks([0.01,0.05,0.09])
    axA[0].set_xticklabels([10,50,90])
    axA[0].set_yticks([0,0.3,0.6,0.9])
        
    axA[0].set_xlabel("Stimulus Intensity (Hz)")
    axA[0].set_ylim([-0.02,0.9])
    axA[0].set_ylabel("Information\nDetection (NMI)")    
    
    
    # axA[0].set_title("Information Detection")
    
    
    # axA[0].annotate('pulling effect', (0.06, .43),
    #             xytext=(0.06, .13), 
    #             arrowprops=dict(arrowstyle="<-, head_width=0.5",lw=2,facecolor='k'),
    #             fontsize=10,
    #             horizontalalignment='center', verticalalignment='top')
    # axA[0].annotate('driving effect', (0.06, 0.45),
    #             xytext=(0.06, .75), 
    #             arrowprops=dict(arrowstyle="<-, head_width=0.5",lw=2,facecolor='k'),
    #             fontsize=10,
    #             horizontalalignment='center', verticalalignment='bottom')
    # axA[0].plot([0.045,0.055,0.055,0.045,0.045],[0.18,0.18,0.65,0.65,0.18],"--k")
    n_column=1
    column="pert"
    pop="pyr"
    method="k_means"
    inpu = 0.05
    
    
    state = "2_2"
    y1 = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][0]
    y1 = 0.9*y1[2]
    state = "6_2"
    y2 = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][0]
    y2 = 1.25*y2[2]
    
    axA[0].arrow(0.055,y1,0,y2-y1,width=0.00001, head_width=0.002, head_length=0.04, fc='k', ec='k')
    axA[0].text(0.6, 0.06, "pulling", transform=axA[0].transAxes,fontsize=7,rotation=90)
    
    
    
    state = "2_2"
    y1 = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][0]
    y1 = 1.05*y1[2]
    state = "2_6"
    y2 = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][0]
    y2 = 0.9*y2[2]
    
    axA[0].arrow(0.055,y1,0,y2-y1,width=0.00001, head_width=0.002, head_length=0.04, fc='k', ec='k')
    
    axA[0].text(0.6, 0.4, "driving", transform=axA[0].transAxes,fontsize=7,rotation=90)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    state = "{}_{}".format(1,1)
    y = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][0,2]
    yerr = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][1,2]
    yerr = t_crit * yerr / np.sqrt(10)
    yerr = yerr
    axA2[0].errorbar(0,y,yerr=yerr, fmt="o",color=colors[state],capsize=capsize, markersize=markersize, capthick=capthick, lw=lw, markeredgewidth=markeredgewidth,alpha=0.8)
    for k in range(len(STATES_ORG)):
        for i,state in enumerate(STATES_ORG[k]):
            y = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][0,2]
            yerr = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][1,2]
            yerr = t_crit * yerr / np.sqrt(10)
            axA2[0].errorbar(1+k+0.15 * (i-1),y,yerr=yerr, fmt="o",color=colors[state],capsize=capsize, markersize=markersize, capthick=capthick, lw=lw, markeredgewidth=markeredgewidth,alpha=0.8)
    axA2[0].set_ylabel("Information\nDetection (NMI)")
    axA2[0].set_xticks([0,1,2,3])
    axA2[0].set_xticklabels(["NREM", "W ("+r"$\bf{LS}$"+")", "W ("+r"$\bf{H}$"+")", "W ("+r"$\bf{DS}$"+")"],rotation=30)
    # axA2[0].set_xticklabels(['\n N',
    #                          r"$\beta_{\mathrm{intra}}>\beta_{\mathrm{inter}}>1$", 
    #                          r"$\beta_{\mathrm{intra}}=\beta_{\mathrm{inter}}>1$", 
    #                          r"$\beta_{\mathrm{inter}}>\beta_{\mathrm{intra}}>1$"], rotation=30)
    # axA2[0].set_title("Information Detection")
    axA2[0].text(0.04, 0.87, "Stimulus=50 Hz", transform=axA2[0].transAxes,fontsize=7)
    
    


    ylimmin,ylimmax = np.inf,-np.inf
    ylim = axA[0].get_ylim()
    ylimmin = min(ylimmin,ylim[0])
    ylimmax = max(ylimmax,ylim[1])
    axA2[0].set_ylim(ylimmin,ylimmax)
    axA2[0].set_yticks([0,0.3,0.6,0.9])

    
    
    
    
    
    
    
    
    state = "{}_{}".format(1,1)
    y = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["diff_{}".format(method)][0][0]
    yerr= data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["diff_{}".format(method)][0][1]
    yerr = t_crit * yerr / np.sqrt(10)
    axB[0].errorbar(0,y,yerr=yerr, fmt="o",color=colors[state],capsize=capsize, markersize=markersize, capthick=capthick, lw=lw, markeredgewidth=markeredgewidth,alpha=0.8)
    for k in range(len(STATES_ORG)):
        for i,state in enumerate(STATES_ORG[k]):
            y= data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["diff_{}".format(method)][0][0]
            yerr = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["diff_{}".format(method)][0][1]
            yerr = t_crit * yerr / np.sqrt(10)
            axB[0].errorbar(1+k+0.15 * (i-1),y,yerr=yerr, fmt="o",color=colors[state],capsize=capsize, markersize=markersize, capthick=capthick, lw=lw, markeredgewidth=markeredgewidth,alpha=0.8)
    axB[0].set_ylabel("Information\nDifferentiation (NMI)")
    axB[0].set_xticks([0,1,2,3])
    axB[0].set_xticklabels(["NREM", "W ("+r"$\bf{LS}$"+")", "W ("+r"$\bf{H}$"+")", "W ("+r"$\bf{DS}$"+")"],rotation=30)
    # axB[0].set_xticklabels(['\n N',
    #                          r"$\beta_{\mathrm{intra}}>\beta_{\mathrm{inter}}>1$", 
    #                          r"$\beta_{\mathrm{intra}}=\beta_{\mathrm{inter}}>1$", 
    #                          r"$\beta_{\mathrm{inter}}>\beta_{\mathrm{intra}}>1$"], rotation=30)
    # axB[0].set_title("Information Differentiation")
    
    
    axB[0].set_yticks([0,0.4,0.8])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    axA[0].text(-0.28,1.15,r"$\bf{a}$",horizontalalignment='left', transform=axA[0].transAxes, fontsize=7)
    
    axA2[0].text(-0.28,1.15,r"$\bf{b}$",horizontalalignment='left', transform=axA2[0].transAxes, fontsize=7)
    
    axB[0].text(-0.28,1.15,r"$\bf{c}$",horizontalalignment='left', transform=axB[0].transAxes, fontsize=7)
    
    
    
    
    fig.savefig(DIR_fig+"fig3.svg")

# 
# 
# 
# 
left=0.07
right=0.002
top=0.16
bottom=0.22


figure_3(left,right,top,bottom)
# 
# 
# 
# 
def figure_4(left,right,top,bottom):
    fix_x = 7.08 #unit in inch
    ws_x = 0.2 #of the axis x lenght
    ws_y = 0.95
    distx = 0.1 #of the axis x lenght
    disty = 0.16
    ax_x = fix_x/(3.3+1*distx+ws_x)
    ax_y = 0.7 * ax_x
    
    fix_y = (2 + 1*disty) * ax_y
    
    gs1 = GridSpec(1,1, bottom=bottom+(1.+1*disty)*ax_y/fix_y, top=-top+1, left=left+0., right=-right+(1)*ax_x/fix_x, wspace=ws_x,hspace = ws_y)
    
    gs2 = GridSpec(2,1, bottom=bottom+0, top=-top+1, left=left+(1.35+distx)*ax_x/fix_x, right=-right+(2.35+distx)*ax_x/fix_x, wspace=ws_x,hspace = ws_y)
    gs3 = GridSpec(2,1, bottom=bottom+0, top=-top+1, left=left+(2.35+distx+ws_x)*ax_x/fix_x, right=-right+1,wspace=ws_x,hspace = ws_y)

    
    fig = plt.figure(figsize=(fix_x, fix_y))
    
    
    axA = []
    axA.append(fig.add_subplot(gs1[0]))
    axB = []
    axC = []
    for i in range(2):
        axB.append(fig.add_subplot(gs2[i]))
        axC.append(fig.add_subplot(gs3[i]))
    
    
    
    
    
    
    
    

    t_crit = -stats.t.ppf(0.05/2,10)


    
    
    
    
    
    axM = [axB,axC]
    
    n_column = 2
    pop="pyr"
    method="k_means"
    
    for ic,column in enumerate(columns):
    
        state = "{}_{}".format(1,1)
        y = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][0,2]
        yerr = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][1,2]
        yerr = t_crit * yerr / np.sqrt(10)
        yerr = yerr
        axM[ic][0].errorbar(0,y,yerr=yerr, fmt="o",color=colors[state],capsize=capsize, markersize=markersize, capthick=capthick, lw=lw, markeredgewidth=markeredgewidth,alpha=0.8)
        for k in range(len(STATES_ORG)):
            for i,state in enumerate(STATES_ORG[k]):
                y = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][0,2]
                yerr = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][1,2]
                yerr = t_crit * yerr / np.sqrt(10)
                axM[ic][0].errorbar(1+k+0.15 * (i-1),y,yerr=yerr, fmt="o",color=colors[state],capsize=capsize, markersize=markersize, capthick=capthick, lw=lw, markeredgewidth=markeredgewidth,alpha=0.8)
        axM[ic][0].set_ylabel("Information\nDetection (NMI)")
        axM[ic][0].set_xticks([0,1,2,3])
        axM[ic][0].set_xticklabels(["NREM", "W ("+r"$\bf{LS}$"+")", "W ("+r"$\bf{H}$"+")", "W ("+r"$\bf{DS}$"+")"],rotation=30)
        # axM[ic][0].set_xticklabels(['\n N',
        #                          r"$\beta_{\mathrm{intra}}>\beta_{\mathrm{inter}}>1$", 
        #                          r"$\beta_{\mathrm{intra}}=\beta_{\mathrm{inter}}>1$", 
        #                          r"$\beta_{\mathrm{inter}}>\beta_{\mathrm{intra}}>1$"], rotation=30)
        axM[ic][0].text(0.04, 0.87, "Stimulus=50 Hz", transform=axM[ic][0].transAxes,fontsize=7)
    
    
    
    
    
        state = "{}_{}".format(1,1)
        y = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["diff_{}".format(method)][0][0]
        yerr = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["diff_{}".format(method)][0][1]
        yerr = t_crit * yerr / np.sqrt(10)
        yerr = yerr
        axM[ic][1].errorbar(0,y,yerr=yerr, fmt="o",color=colors[state],capsize=capsize, markersize=markersize, capthick=capthick, lw=lw, markeredgewidth=markeredgewidth,alpha=0.8)
        for k in range(len(STATES_ORG)):
            for i,state in enumerate(STATES_ORG[k]):
                y= data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["diff_{}".format(method)][0][0]
                yerr = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["diff_{}".format(method)][0][1]
                yerr = t_crit * yerr / np.sqrt(10)
                axM[ic][1].errorbar(1+k+0.15 * (i-1),y,yerr=yerr, fmt="o",color=colors[state],capsize=capsize, markersize=markersize, capthick=capthick, lw=lw, markeredgewidth=markeredgewidth,alpha=0.8)
        axM[ic][1].set_ylabel("Information\nDifferentiation (NMI)")
        axM[ic][1].set_xticks([0,1,2,3])
        axM[ic][1].set_xticklabels(["NREM", "W ("+r"$\bf{LS}$"+")", "W ("+r"$\bf{H}$"+")", "W ("+r"$\bf{DS}$"+")"],rotation=30)
    
    
    
    
    
    for i in range(2):
        axM[0][i].set_title("Perturbed Column")
        axM[1][i].set_title("Unperturbed Column")
    
    
    
    
    
    
    
    
    

    
    
    
    
    axB[0].set_yticks([0.2,0.6,1])
    axB[1].set_yticks([0.,0.4,0.8])
    
    
    axC[0].set_yticks([0.,0.07,0.14])
    # axB[1].set_yticks([0.,0.4,0.8])
    
    
    
    
    
    lines = [Line2D([0], [0], color="white", marker='o',markersize=0.005, markerfacecolor="white") for c in range(5)]
    labels  = ["Intra-Excitatory\nConnections",
               "Inter-Excitatory\nConnections",
              "Inhibitory\nConnections","Noise","Stimulus"]
    
    # axA[0].legend(lines, labels,handlelength=3,loc='center right', bbox_to_anchor=(1.1, 0.5),fontsize=7)
    
    
    
    
    
    
    axA[0].text(-0.24,1.15,r"$\bf{a}$",horizontalalignment='left', transform=axA[0].transAxes, fontsize=7)
    axA[0].axis('off')
    
    axB[0].text(-0.35,1.15,r"$\bf{b}$"+"(i)",horizontalalignment='left', transform=axB[0].transAxes, fontsize=7)
    axB[1].text(-0.33,1.15,"(ii)",horizontalalignment='left', transform=axB[1].transAxes, fontsize=7)
    
    axC[0].text(-0.41,1.15,r"$\bf{c}$"+"(i)",horizontalalignment='left', transform=axC[0].transAxes, fontsize=7)
    axC[1].text(-0.39,1.15,"(ii)",horizontalalignment='left', transform=axC[1].transAxes, fontsize=7)
    
    
    
    
    
    
    fig.savefig(DIR_fig+"fig4.svg")
# 
# 
# 
# 
left=0.059
right=0.005
top=0.08
bottom=0.11


figure_4(left,right,top,bottom)
# 
# 
# 
# 
# 
# 
# 
# 
def figure_S_1(left,right,top,bottom):
    fix_x = 7.08#unit in inch
    ws_x = 0.34#of the axis x lenght
    ws_y = 0.7
    distx = 0.48 #of the axis x lenght
    disty = 0.09
    ax_x = fix_x/(3+2*ws_x)
    ax_y = 0.75 * ax_x
    
    fix_y = (4. + 3*disty) * ax_y
    
    gs1 = GridSpec(1,3, bottom=bottom+(3+3*disty)*ax_y/fix_y, top=-top+1, left=left+0, right=-right+1,wspace=ws_x,hspace = ws_y)
    gs2 = GridSpec(1,3, bottom=bottom+(2+2*disty)*ax_y/fix_y, top=-top+(3+2*disty)*ax_y/fix_y, left=left+0, right=-right+1,wspace=ws_x,hspace = ws_y)
    gs3 = GridSpec(1,3, bottom=bottom+(1+disty)*ax_y/fix_y, top=-top+(2.+disty)*ax_y/fix_y, left=left+0, right=-right+1,wspace=ws_x,hspace = ws_y)
    gs4 = GridSpec(1,3, bottom=bottom+0, top=-top+(1)*ax_y/fix_y, left=left+0, right=-right+1,wspace=ws_x,hspace = ws_y)
    
    fig = plt.figure(figsize=(fix_x, fix_y))
    
    
    axA = []
    axB = []
    axC = []
    axD = []
    for i in range(3):
        axA.append(fig.add_subplot(gs1[i]))
        axB.append(fig.add_subplot(gs2[i]))
        axC.append(fig.add_subplot(gs3[i]))
        axD.append(fig.add_subplot(gs4[i]))
    ax_com = [axA,axB,axC,axD]
    
    
    
    
    
    
    
    
    
    
    
    bimodal_limit = -0.014, 0.265
    power_limit = 0,0.715
    
    for i,std in enumerate([0.9, 1.1,1.2,1.4]):
        if std !=1.2:
            ax_com[i][0].plot(data["prestimulus_std_analysis"]['std:{}'.format(std)]["instance"][::10],color="gray")
        else:
            ax_com[i][0].plot(data["prestimulus_spontaneous_analysis"]["1_1"]["1_column"]["instance"][::10],color=colors["1_1"],alpha=0.8)
        ax_com[i][0].set_xlabel("Time (s)")
        ax_com[i][0].set_xticks([0,2000,4000])
        ax_com[i][0].set_xticklabels([0,2,4])
        ax_com[i][0].set_ylim(0,30)
        ax_com[i][0].set_ylabel(r"$\phi$"+" = {} ms".format(std)+r"$^{-1}$"+"\n\nFiring Rate (Hz)")
    
    
        ylimmin,ylimmax=np.inf,-np.inf
        
        hist = data["prestimulus_std_analysis"]['std:{}'.format(std)]["hist"]/100
        hist_std = data["prestimulus_std_analysis"]['std:{}'.format(std)]["hist_std"]/100
        hist_bin = data["prestimulus_std_analysis"]['std:{}'.format(std)]["hist_bin"]
    
        if std !=1.2:
            ax_com[i][1].plot(hist_bin,hist,color="gray")
            ax_com[i][1].fill_between(hist_bin,hist-hist_std,hist+hist_std,color="gray",alpha=0.2)
        else:
            hist = data["prestimulus_spontaneous_analysis"]["1_1"]["1_column"]["hist"]/100
            hist_std = data["prestimulus_spontaneous_analysis"]["1_1"]["1_column"]["hist_std"]/100
            hist_bin = data["prestimulus_spontaneous_analysis"]["1_1"]["1_column"]["hist_bin"]
            
            ax_com[i][1].plot(hist_bin,hist,color=colors["1_1"],alpha=0.8)
            ax_com[i][1].fill_between(hist_bin,hist-hist_std,hist+hist_std,color=colors["1_1"],alpha=0.2)
    
        ylim = ax_com[i][1].get_ylim()
        ylimmin = min(ylimmin,ylim[0])
        ylimmax = max(ylimmax,ylim[1])
        ax_com[i][1].set_xlabel("Firing Rate (Hz)")
    
    for i in range(4):
        ax_com[i][1].set_ylim(bimodal_limit)
    
    
    axinylimmin,axinylimmax=np.inf,-np.inf
    ylimmin,ylimmax=np.inf,-np.inf
    for i,std in enumerate([0.9, 1.1,1.2,1.4]):
        rel_pow = data["prestimulus_std_analysis"]['std:{}'.format(std)]["rel_pow"]
        rel_pow_std = data["prestimulus_std_analysis"]['std:{}'.format(std)]["rel_pow_std"]
    
        if std !=1.2:
            ax_com[i][2].bar(np.arange(len(rel_pow)), rel_pow, yerr=rel_pow_std, align='center', alpha=0.8,color="gray", ecolor='black', capsize=2, error_kw={"elinewidth":0.8,"capthick": 0.8})
        else:
            rel_pow = data["prestimulus_spontaneous_analysis"]["1_1"]["1_column"]["rel_pow"]
            rel_pow_std = data["prestimulus_spontaneous_analysis"]["1_1"]["1_column"]["rel_pow_std"]
            ax_com[i][2].bar(np.arange(len(rel_pow)), rel_pow, yerr=rel_pow_std, align='center', alpha=0.8,color=colors["1_1"], ecolor='black', capsize=2, error_kw={"elinewidth":0.8,"capthick": 0.8})
        
        ax_com[i][2].set_xticks(np.arange(len(rel_pow)))
        ax_com[i][2].set_xticklabels(["SO","Delta","Theta","Alpha","Beta","Gamma"],rotation = 45)
    
        ylim = ax_com[i][2].get_ylim()
        ylimmin = min(ylimmin,ylim[0])
        ylimmax = max(ylimmax,ylim[1])
    
    
        axinylimmin = min(axinylimmin,rel_pow[-1]-rel_pow_std[-1])
        axinylimmax = max(axinylimmax,rel_pow[-1]+rel_pow_std[-1])
    
    # power_limit = 0,0.7
    for i in range(4):
        ax_com[i][2].set_ylim(power_limit) 
    
        ax_com[i][1].set_ylabel("Occurrence")
    
        ax_com[i][2].set_ylabel("Power")
    
    axA[0].set_title("One Representative Trial")
    axA[1].set_title("Histogram")
    axA[2].set_title("Power Spectrum")
    
    
    
    
    
    
    axA[0].text(-0.325,1.15,r"$\bf{a}$"+"(i)",horizontalalignment='left', transform=axA[0].transAxes, fontsize=7)
    axB[0].text(-0.325,1.15,r"$\bf{b}$"+"(i)",horizontalalignment='left', transform=axB[0].transAxes, fontsize=7)
    axC[0].text(-0.325,1.15,r"$\bf{c}$"+"(i)",horizontalalignment='left', transform=axC[0].transAxes, fontsize=7)
    axD[0].text(-0.325,1.15,r"$\bf{d}$"+"(i)",horizontalalignment='left', transform=axD[0].transAxes, fontsize=7)
    
    for j in range(4):
        ax_com[j][1].text(-0.23,1.15,"(ii)",horizontalalignment='left', transform=ax_com[j][1].transAxes, fontsize=7)
        ax_com[j][2].text(-0.23,1.15,"(iii)",horizontalalignment='left', transform=ax_com[j][2].transAxes, fontsize=7)
    
    
    
    fig.savefig(DIR_fig+"figs1.svg")
# 
# 
# 
# 

left=0.084
right=0.002
top=0.036
bottom=0.062


figure_S_1(left,right,top,bottom)
# 
# 
# 
# 
def figure_S_2(left,right,top,bottom):
    fix_x = 7.08#unit in inch
    ws_x = 0.34#of the axis x lenght
    ws_y = 0.7
    distx = 0.48 #of the axis x lenght
    disty = 0.0
    ax_x = fix_x/(3+ws_x+distx)
    ax_y = 0.8 * ax_x
    
    fix_y = (6. + 5*disty) * ax_y
    
    gs1 = GridSpec(1,3, bottom=bottom+(5+5*disty)*ax_y/fix_y, top=-top+1, left=left+0, right=-right+1,wspace=ws_x,hspace = ws_y)
    gs2 = GridSpec(1,3, bottom=bottom+(4+4*disty)*ax_y/fix_y, top=-top+(5+4*disty)*ax_y/fix_y, left=left+0, right=-right+1,wspace=ws_x,hspace = ws_y)
    gs3 = GridSpec(1,3, bottom=bottom+(3+3*disty)*ax_y/fix_y, top=-top+(4+3*disty)*ax_y/fix_y, left=left+0, right=-right+1,wspace=ws_x,hspace = ws_y)
    gs4 = GridSpec(1,3, bottom=bottom+(2+2*disty)*ax_y/fix_y, top=-top+(3+2*disty)*ax_y/fix_y, left=left+0, right=-right+1,wspace=ws_x,hspace = ws_y)
    gs5 = GridSpec(1,3, bottom=bottom+(1+disty)*ax_y/fix_y, top=-top+(2.+disty)*ax_y/fix_y, left=left+0, right=-right+1,wspace=ws_x,hspace = ws_y)
    gs6 = GridSpec(1,3, bottom=bottom+0, top=-top+(1)*ax_y/fix_y, left=left+0, right=-right+1,wspace=ws_x,hspace = ws_y)
    
    fig = plt.figure(figsize=(fix_x, fix_y))
    
    
    axA = []
    axB = []
    axC = []
    axD = []
    axE = []
    axF = []
    for i in range(3):
        axA.append(fig.add_subplot(gs1[i]))
        axB.append(fig.add_subplot(gs2[i]))
        axC.append(fig.add_subplot(gs3[i]))
        axD.append(fig.add_subplot(gs4[i]))
        axE.append(fig.add_subplot(gs5[i]))
        axF.append(fig.add_subplot(gs6[i]))
    ax_com = [axB,axC,axD,axE,axF]
    
    
    
    
    
    
    
    
    
    bimodal_limit = -0.014, 0.265
    power_limit = 0,0.715
    
    
    
    
    
    
    
    state = "1_1"
    axA[0].plot(data["prestimulus_spontaneous_analysis"][state]["1_column"]["instance"][::10],color=colors[state],alpha=0.8)
    
    axA[0].set_xlabel("Time (s)")
    axA[0].set_xticks([0,2000,4000])
    axA[0].set_xticklabels([0,2,4])
    axA[0].set_ylim(0,30)
    axA[0].set_ylabel(r"$\beta_{\mathrm{intra}}=$"+"{}".format(1)+"\n\nFiring Rate (Hz)")
    
    
    ylimmin,ylimmax=np.inf,-np.inf
    
    hist = data["prestimulus_spontaneous_analysis"][state]["1_column"]["hist"]/100
    hist_std = data["prestimulus_spontaneous_analysis"][state]["1_column"]["hist_std"]/100
    hist_bin = data["prestimulus_spontaneous_analysis"][state]["1_column"]["hist_bin"]
    
    axA[1].plot(hist_bin,hist,color=colors[state],alpha=0.8)
    axA[1].fill_between(hist_bin,hist-hist_std,hist+hist_std,color=colors[state],alpha=0.2)
    
    ylim = axA[1].get_ylim()
    ylimmin = min(ylimmin,ylim[0])
    ylimmax = max(ylimmax,ylim[1])
    axA[1].set_xlabel("Firing Rate (Hz)")
    
    axA[1].set_ylim(bimodal_limit)
    
    
    
    rel_pow = data["prestimulus_spontaneous_analysis"][state]["1_column"]["rel_pow"]
    rel_pow_std = data["prestimulus_spontaneous_analysis"][state]["1_column"]["rel_pow_std"]
    axA[2].bar(np.arange(len(rel_pow)), rel_pow, yerr=rel_pow_std, align='center', alpha=0.8,color=colors[state], ecolor='black', capsize=2, error_kw={"elinewidth":0.8,"capthick": 0.8})
    axA[2].set_xticks(np.arange(len(rel_pow)))
    axA[2].set_xticklabels(["SO","Delta","Theta","Alpha","Beta","Gamma"],rotation = 45)
    
    ylim = axA[2].get_ylim()
    ylimmin = min(ylimmin,ylim[0])
    ylimmax = max(ylimmax,ylim[1])
    
    
    
    
    axA[2].set_ylim(power_limit) 
    
    axA[1].set_ylabel("Occurrence")
    
    axA[2].set_ylabel("Power")
    
    axA[0].set_title("One Representative Trial")
    axA[1].set_title("Histogram")
    axA[2].set_title("Power Spectrum")
    
    



    state = "2_2"
    axD[0].plot(data["prestimulus_spontaneous_analysis"][state]["1_column"]["instance"][::10],color=colors[state],alpha=0.8)
    
    
    hist = data["prestimulus_spontaneous_analysis"][state]["1_column"]["hist"]/100
    hist_std = data["prestimulus_spontaneous_analysis"][state]["1_column"]["hist_std"]/100
    hist_bin = data["prestimulus_spontaneous_analysis"][state]["1_column"]["hist_bin"]
    
    axD[1].plot(hist_bin,hist,color=colors[state],alpha=0.8)
    axD[1].fill_between(hist_bin,hist-hist_std,hist+hist_std,color=colors[state],alpha=0.2)
    
    
    
    
    rel_pow = data["prestimulus_spontaneous_analysis"][state]["1_column"]["rel_pow"]
    rel_pow_std = data["prestimulus_spontaneous_analysis"][state]["1_column"]["rel_pow_std"]
    axD[2].bar(np.arange(len(rel_pow)), rel_pow, yerr=rel_pow_std, align='center', alpha=0.8,color=colors[state], ecolor='black', capsize=2, error_kw={"elinewidth":0.8,"capthick": 0.8})
    # axB[2].set_xticks(np.arange(len(rel_pow)))
    # axB[2].set_xticklabels(["SO","Delta","Theta","Alpha","Beta","Gamma"],rotation = 45)
    
    




    
    
    
    
    
    for i,beta in enumerate([1.2,1.6,2.,4.,6.]):
        state = "{}_2".format(int(beta))
        if beta in [1.2,1.6]:
            ax_com[i][0].plot(data["prestimulus_beta_analysis"]['beta:{}'.format(beta)]["instance"][::10],color="gray")
        elif beta in [4.,6.]:
            ax_com[i][0].plot(data["prestimulus_beta_analysis"]['beta:{}'.format(beta)]["instance"][::10],color=colors[state],alpha=0.8)
        
    
            
        ax_com[i][0].set_xlabel("Time (s)")
        ax_com[i][0].set_xticks([0,2000,4000])
        ax_com[i][0].set_xticklabels([0,2,4])
        ax_com[i][0].set_ylim(0,30)
        ax_com[i][0].set_ylabel(r"$\beta_{\mathrm{intra}}=$"+"{}".format(beta)+"\n\nFiring Rate (Hz)")
        if i > 1:
            ax_com[i][0].set_ylabel(r"$\beta_{\mathrm{intra}}=$"+"{}".format(int(beta))+"\n\nFiring Rate (Hz)")
    
        ylimmin,ylimmax=np.inf,-np.inf
        
        hist = data["prestimulus_beta_analysis"]['beta:{}'.format(beta)]["hist"]/100
        hist_std = data["prestimulus_beta_analysis"]['beta:{}'.format(beta)]["hist_std"]/100
        hist_bin = data["prestimulus_beta_analysis"]['beta:{}'.format(beta)]["hist_bin"]
    
        if beta in [1.2,1.6]:
            ax_com[i][1].plot(hist_bin,hist,color="gray")
            ax_com[i][1].fill_between(hist_bin,hist-hist_std,hist+hist_std,color="gray",alpha=0.2)
        elif beta in [4.,6.]:
            ax_com[i][1].plot(hist_bin,hist,color=colors[state],alpha=0.8)
            ax_com[i][1].fill_between(hist_bin,hist-hist_std,hist+hist_std,color=colors[state],alpha=0.2)
    
        ylim = ax_com[i][1].get_ylim()
        ylimmin = min(ylimmin,ylim[0])
        ylimmax = max(ylimmax,ylim[1])
        ax_com[i][1].set_xlabel("Firing Rate (Hz)")
    
    for i in range(4):
        ax_com[i][1].set_ylim(bimodal_limit)
    
    
    axinylimmin,axinylimmax=np.inf,-np.inf
    ylimmin,ylimmax=np.inf,-np.inf
    for i,beta in enumerate([1.2,1.6,2.,4.,6.]):
        state = "{}_2".format(int(beta))
        rel_pow = data["prestimulus_beta_analysis"]['beta:{}'.format(beta)]["rel_pow"]
        rel_pow_std = data["prestimulus_beta_analysis"]['beta:{}'.format(beta)]["rel_pow_std"]
        if beta in [1.2,1.6]:
            ax_com[i][2].bar(np.arange(len(rel_pow)), rel_pow, yerr=rel_pow_std, align='center', alpha=0.8,color="gray", ecolor='black', capsize=2, error_kw={"elinewidth":0.8,"capthick": 0.8})
        elif beta in [4.,6.]:
            ax_com[i][2].bar(np.arange(len(rel_pow)), rel_pow, yerr=rel_pow_std, align='center', alpha=0.8,color=colors[state], ecolor='black', capsize=2, error_kw={"elinewidth":0.8,"capthick": 0.8})
        
        ax_com[i][2].set_xticks(np.arange(len(rel_pow)))
        ax_com[i][2].set_xticklabels(["SO","Delta","Theta","Alpha","Beta","Gamma"],rotation = 45)
    
        ylim = ax_com[i][2].get_ylim()
        ylimmin = min(ylimmin,ylim[0])
        ylimmax = max(ylimmax,ylim[1])
    
    
    
    # power_limit = 0,0.7
    for i in range(5):
        ax_com[i][2].set_ylim(power_limit) 
    
        ax_com[i][1].set_ylabel("Occurrence")
    
        ax_com[i][2].set_ylabel("Power")
    
    
    
    
    
    
    
    
    axA[0].text(-0.325,1.15,r"$\bf{a}$"+"(i)",horizontalalignment='left', transform=axA[0].transAxes, fontsize=7)
    axB[0].text(-0.325,1.15,r"$\bf{b}$"+"(i)",horizontalalignment='left', transform=axB[0].transAxes, fontsize=7)
    axC[0].text(-0.325,1.15,r"$\bf{c}$"+"(i)",horizontalalignment='left', transform=axC[0].transAxes, fontsize=7)
    axD[0].text(-0.325,1.15,r"$\bf{d}$"+"(i)",horizontalalignment='left', transform=axD[0].transAxes, fontsize=7)
    axE[0].text(-0.325,1.15,r"$\bf{e}$"+"(i)",horizontalalignment='left', transform=axE[0].transAxes, fontsize=7)
    axF[0].text(-0.325,1.15,r"$\bf{f}$"+"(i)",horizontalalignment='left', transform=axF[0].transAxes, fontsize=7)

    axA[0].text(-0.23,1.15,"(ii)",horizontalalignment='left', transform=axA[1].transAxes, fontsize=7)
    axA[0].text(-0.23,1.15,"(iii)",horizontalalignment='left', transform=axA[2].transAxes, fontsize=7)
    
    for j in range(5):
        ax_com[j][1].text(-0.23,1.15,"(ii)",horizontalalignment='left', transform=ax_com[j][1].transAxes, fontsize=7)
        ax_com[j][2].text(-0.23,1.15,"(iii)",horizontalalignment='left', transform=ax_com[j][2].transAxes, fontsize=7)
    
    
    
    fig.savefig(DIR_fig+"figs2.svg")
# 
# 
# 
# 
left=0.084
right=0.002
top=0.025
bottom=0.044




figure_S_2(left,right,top,bottom)
# 
# 
# 
# 
def figure_S_3(left,right,top,bottom):
    fix_x = 7.08#unit in inch
    ws_x = 0.6 #of the axis x lenght
    ws_y = 0.8
    distx = 0.48 #of the axis x lenght
    disty = 0.8
    ax_x = fix_x/(3+2*ws_x)
    ax_y = 0.8 * ax_x
    
    fix_y = (2+ws_y) * ax_y
    
    
    gs1 = GridSpec(2,3, bottom=bottom, top=-top+1, left=left, right=-right+1,wspace=ws_x,hspace = ws_y)
    
    
    fig = plt.figure(figsize=(fix_x, fix_y))
    
    
    ax= []
    for i in range(2):
        for j in range(3):
            ax.append(fig.add_subplot(gs1[i,j]))
    
    ax = np.array(ax).reshape(2,3)
    
    gs = ax[1, 1].get_gridspec()
    # remove the underlying axes
    for axs in ax[1, 1:]:
        axs.remove()
    axbig = fig.add_subplot(gs[1,1:])
    
    
    for state in ["2_2","2_6","6_2"]:
        ax[0,0].plot(data["deterministic_time_trace_E_I_one_column_input_0.05"][state],c=colors[state],alpha=0.8,markersize=markersize, zorder=2)
    
    ax[0,0].set_xlim(9500,15000)
    ax[0,0].set_xticks([10000,14000])
    ax[0,0].set_xticklabels([0,0.4])
    
    ax[0,0].set_xlabel("Time (s)")
    ax[0,0].set_ylabel(r'$|\mathrm{E}|-|\mathrm{I}|$'+" (mV)")
    
    
    
    for state in ["2_2","2_6","6_2"]:
        ei_aux=data["deterministic_response"]["{}".format(state)]["1_column"]["input:0.05"]["pert"]["pyr"]['firing'][0]
        ax[1,0].plot(ei_aux, color=colors[state], label='Curve 2',alpha=0.8,markersize=markersize, zorder=2)
    
    ax[1,0].set_yticks([22,25,28])
    ax[1,0].set_xlim(9500,15000)
    ax[1,0].set_xticks([10000,14000])
    ax[1,0].set_xticklabels([0,0.4])
    
    # Customize inset plot
    ax[1,0].set_xlabel('Time (s)')
    ax[1,0].set_ylabel('Firing Rate (Hz)')
    
    
    
    
    
    ei = np.zeros((3,3),dtype=float)
    for i,bintra in enumerate([2,4,6]):
        for j,binter in enumerate([2,4,6]):
            ei_aux=data["deterministic_response"]["{}_{}".format(bintra,binter)]["1_column"]["input:0.05"]["pert"]["pyr"]['E/I_post']
            ei_aux = ei_aux[0]+ei_aux[1]+ei_aux[3]-ei_aux[2]
            ei[j,i] = ei_aux
    
    g1 = sns.heatmap(ei, cmap = 'gray_r',center=0.98*(ei.max() + ei.min())/2, cbar_kws={'label': r'$|\mathrm{E}|-|\mathrm{I}|$'+" (mV)",'aspect': 10},ax=ax[0,1])
    g1.invert_yaxis()
    g1.set_xlabel(r"$\beta_{\mathrm{intra}}$")
    g1.set_ylabel(r"$\beta_{\mathrm{inter}}$")
    g1.set_yticks([0.5,1.5,2.5])
    g1.set_yticklabels([2,4,6])
    
    g1.set_xticks([0.5,1.5,2.5])
    g1.set_xticklabels([2,4,6])
    
    
    
    
    ei = np.zeros((3,3),dtype=float)
    for i,bintra in enumerate([2,4,6]):
        for j,binter in enumerate([2,4,6]):
            ei_aux=data["deterministic_response"]["{}_{}".format(bintra,binter)]["1_column"]["input:0.05"]["pert"]["pyr"]['firing']
            ei_aux = ei_aux[0,11000] - ei_aux[0,9000]
            ei[j,i] = ei_aux
    
    
    g2 = sns.heatmap(ei, cmap = 'gray_r',center=0.98*(ei.max() + ei.min())/2, cbar_kws={'label': 'Amplitute (Hz)','aspect': 10},ax=ax[0,2])
    g2.invert_yaxis()
    g2.set_xlabel(r"$\beta_{\mathrm{intra}}$")
    g2.set_ylabel(r"$\beta_{\mathrm{inter}}$")
    g2.set_yticks([0.5,1.5,2.5])
    g2.set_yticklabels([2,4,6])
    
    g2.set_xticks([0.5,1.5,2.5])
    g2.set_xticklabels([2,4,6])
    
    
    
    for i,bintra in enumerate([2,4,6]):
        for j,binter in enumerate([2,4,6]):
            ei_aux=data["deterministic_response"]["{}_{}".format(bintra,binter)]["1_column"]["input:0.05"]["pert"]["pyr"]['E/I_post']
            ei_aux = ei_aux[0]+ei_aux[1]+ei_aux[3]-ei_aux[2]
    
    
            ei_aux_1=data["deterministic_response"]["{}_{}".format(bintra,binter)]["1_column"]["input:0.05"]["pert"]["pyr"]['firing']
            ei_aux_1 = ei_aux_1[0,11000] - ei_aux_1[0,9000]
    
    
            state = "{}_{}".format(bintra,binter)
            axbig.plot(ei_aux,ei_aux_1,"o",c=colors[state],alpha=0.8,markersize=markersize)
    
    axbig.text(0.04, 0.87, "Stimulus=50 Hz", transform=axbig.transAxes,fontsize=7)
    axbig.set_xlabel(r'$|\mathrm{E}|-|\mathrm{I}|$'+" (mV)")
    axbig.set_ylabel("Amplitude (Hz)")
    
    
    for i in range(2):
        ylimmin,ylimmax = np.inf,-np.inf
        ylim = ax[i,0].get_ylim()
    
        ylimmin = min(ylimmin,ylim[0])
        ylimmax = max(ylimmax,ylim[1])
        
        ax[i,0].fill_between([10000,11000],[ylimmin]*2,[ylimmax+0.04*(ylimmax-ylimmin)]*2,color="gray",alpha=0.3, linewidth=0, zorder=1)
        ax[i,0].set_ylim(ylimmin,ylimmax)
        
        
    ax[0,0].text(0.48, 0.87, "Stimulus=50 Hz", transform=ax[0,0].transAxes,fontsize=7)
    ax[1,0].text(0.48, 0.87, "Stimulus=50 Hz", transform=ax[1,0].transAxes,fontsize=7)
    
    ax[0,1].set_title("Stimulus=50 Hz")
    ax[0,2].set_title("Stimulus=50 Hz")   
    ax[0,0].text(-0.29,1.15,r"$\bf{a}$"+"(i)",horizontalalignment='left', transform=ax[0,0].transAxes, fontsize=7)
    ax[1,0].text(-0.25,1.15,"(ii)",horizontalalignment='left', transform=ax[1,0].transAxes, fontsize=7)
    ax[0,1].text(-0.30,1.15,r"$\bf{b}$"+"(i)",horizontalalignment='left', transform=ax[0,1].transAxes, fontsize=7)
    ax[0,2].text(-0.28,1.15,"(ii)",horizontalalignment='left', transform=ax[0,2].transAxes, fontsize=7)
    axbig.text(-0.08,1.15,"(iii)",horizontalalignment='left', transform=axbig.transAxes, fontsize=7)
    
    fig.align_ylabels(ax[:,0])
    
    
    fig.savefig(DIR_fig+"figs3.svg")
# 
# 
# 
# 
left=0.065
right=0.03
top=0.07
bottom=0.1


figure_S_3(left,right,top,bottom)
# 
# 
# 
# 
def figure_S_4(left,right,top,bottom):   
    fix_x = 7.08#unit in inch
    ws_x = 0.15 #of the axis x lenght
    ws_y = 0.6
    distx = 0.48 #of the axis x lenght
    disty = 0.7
    ax_x = fix_x/(4+3*ws_x)
    ax_y = 0.8 * ax_x
    
    fix_y = (1) * ax_y
    
    
    gs3 = GridSpec(1,4, bottom=bottom+0, top=-top+1, left=left+0, right=-right+1,wspace=ws_x,hspace = ws_y)
    
    fig = plt.figure(figsize=(fix_x, fix_y))
    
    
    
    ax = []
    for i in range(4):
        ax.append(fig.add_subplot(gs3[i]))
    
    
    
    
    
    ylimmin,ylimmax = np.inf,-np.inf
    for ij,inpu in enumerate([0.01,0.03,0.07,0.09]):
        
        if inpu!=0.05:
            state = "{}_{}".format(1,1)
            y = data['deterministic_response'][state]["1_column"]["input:{}".format(inpu)]["pert"]["pyr"]["firing"][0,11000]
            y = y - data['deterministic_response'][state]["1_column"]["input:{}".format(inpu)]["pert"]["pyr"]["firing"][0,9000]
            ax[ij].plot(0,y, "o",color=colors[state],alpha=0.8,markersize=markersize)
            for k in range(len(STATES_ORG)):
                for i,state in enumerate(STATES_ORG[k]):
                    y = data['deterministic_response'][state]["1_column"]["input:{}".format(inpu)]["pert"]["pyr"]["firing"][0,11000]
                    y = y - data['deterministic_response'][state]["1_column"]["input:{}".format(inpu)]["pert"]["pyr"]["firing"][0,9000]
                    ax[ij].plot(1+k+0.15 * (i-1),y, "o",color=colors[state],alpha=0.8,markersize=markersize)
            
            ax[ij].set_xticks([0,1,2,3])
            ax[ij].set_xticklabels(["NREM", "W ("+r"$\bf{LS}$"+")", "W ("+r"$\bf{H}$"+")", "W ("+r"$\bf{DS}$"+")"],rotation=30)
            # ax[ij].set_xticklabels(['\n N',
            #                          r"$\beta_{\mathrm{intra}}>\beta_{\mathrm{inter}}>1$", 
            #                          r"$\beta_{\mathrm{intra}}=\beta_{\mathrm{inter}}>1$", 
            #                          r"$\beta_{\mathrm{inter}}>\beta_{\mathrm{intra}}>1$"], rotation=30)
            # ax[ij].set_title("Information Detection")
            ylim = ax[ij].get_ylim()
            ylimmin = min(ylimmin,ylim[0])
            ylimmax = max(ylimmax,ylim[1])
            
            ax[ij].text(0.2, 0.87, "Stimulus={} Hz".format(int(1000*inpu)), transform=ax[ij].transAxes,fontsize=7)
        
    for i in range(4):
        ax[i].set_ylim(ylimmin,ylimmax)
        ax[i].set_yticks([0,5,10])
        ax[i].set_yticklabels([])
    ax[0].set_yticklabels([0,5,10])
    ax[0].set_ylabel("Amplitude (Hz)")
    
    
    
    
    ax[0].text(-0.24,1.15,r"$\bf{a}$",horizontalalignment='left', transform=ax[0].transAxes, fontsize=7)
    ax[1].text(-0.07,1.15,r"$\bf{b}$",horizontalalignment='left', transform=ax[1].transAxes, fontsize=7)
    ax[2].text(-0.07,1.15,r"$\bf{c}$",horizontalalignment='left', transform=ax[2].transAxes, fontsize=7)
    ax[3].text(-0.07,1.15,r"$\bf{d}$",horizontalalignment='left', transform=ax[3].transAxes, fontsize=7)
    
    
    
    
    
    fig.savefig(DIR_fig+"figs4.svg")
# 
# 
# 
# 
left=0.055
right=0.0029
top=0.17
bottom=0.27


figure_S_4(left,right,top,bottom)
# 
# 
# 
# 
def figure_S_5(left,right,top,bottom):   
    fix_x = 7.08#unit in inch
    ws_x = 0.15 #of the axis x lenght
    ws_y = 0.6
    distx = 0.48 #of the axis x lenght
    disty = 0.7
    ax_x = fix_x/(4+3*ws_x)
    ax_y = 0.8 * ax_x
    
    fix_y = (1) * ax_y
    
    
    gs3 = GridSpec(1,4, bottom=bottom+0, top=-top+1, left=left+0, right=-right+1,wspace=ws_x,hspace = ws_y)
    
    fig = plt.figure(figsize=(fix_x, fix_y))
    
    
    
    ax = []
    for i in range(4):
        ax.append(fig.add_subplot(gs3[i]))
    
    
    n_column = 1
    column = "pert"
    pop = "pyr"
    method = "k_means"
    
    
    t_crit = -stats.t.ppf(0.05/2,10)
    
    

    ylimmin,ylimmax = np.inf,-np.inf
    for ij,inpu in enumerate([0.01,0.03,0.07,0.09]):
        
        if inpu!=0.05:
            state = "{}_{}".format(1,1)
            y = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][0,ij]
            yerr = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][1,ij]
            yerr = t_crit * yerr / np.sqrt(10)
            yerr = yerr
            ax[ij].errorbar(0,y,yerr=yerr, fmt="o",color=colors[state],capsize=capsize, markersize=markersize, capthick=capthick, lw=lw, markeredgewidth=markeredgewidth,alpha=0.8)
            for k in range(len(STATES_ORG)):
                for i,state in enumerate(STATES_ORG[k]):
                    y = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][0,ij]
                    yerr = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][1,ij]
                    yerr = t_crit * yerr / np.sqrt(10)
                    ax[ij].errorbar(1+k+0.15 * (i-1),y,yerr=yerr, fmt="o",color=colors[state],capsize=capsize, markersize=markersize, capthick=capthick, lw=lw, markeredgewidth=markeredgewidth,alpha=0.8)
            
            ax[ij].set_xticks([0,1,2,3])
            ax[ij].set_xticklabels(["NREM", "W ("+r"$\bf{LS}$"+")", "W ("+r"$\bf{H}$"+")", "W ("+r"$\bf{DS}$"+")"],rotation=30)
            # ax[ij].set_xticklabels(['\n N',
            #                          r"$\beta_{\mathrm{intra}}>\beta_{\mathrm{inter}}>1$", 
            #                          r"$\beta_{\mathrm{intra}}=\beta_{\mathrm{inter}}>1$", 
            #                          r"$\beta_{\mathrm{inter}}>\beta_{\mathrm{intra}}>1$"], rotation=30)
            # ax[ij].set_title("Information Detection")
            ylim = ax[ij].get_ylim()
            ylimmin = min(ylimmin,ylim[0])
            ylimmax = max(ylimmax,ylim[1])
            
            ax[ij].text(0.04, 0.87, "Stimulus={} Hz".format(int(1000*inpu)), transform=ax[ij].transAxes,fontsize=7)
        
    for i in range(4):
        ax[i].set_ylim(ylimmin,ylimmax)
        ax[i].set_yticks([0,0.4,0.8])
        ax[i].set_yticklabels([])
    ax[0].set_yticklabels([0,0.4,0.8])
    ax[0].set_ylabel("Information\nDetection (NMI)")
    
    
    
    ax[0].text(-0.32,1.15,r"$\bf{a}$",horizontalalignment='left', transform=ax[0].transAxes, fontsize=7)
    ax[1].text(-0.07,1.15,r"$\bf{b}$",horizontalalignment='left', transform=ax[1].transAxes, fontsize=7)
    ax[2].text(-0.07,1.15,r"$\bf{c}$",horizontalalignment='left', transform=ax[2].transAxes, fontsize=7)
    ax[3].text(-0.07,1.15,r"$\bf{d}$",horizontalalignment='left', transform=ax[3].transAxes, fontsize=7)
    
    
    
    
    
    fig.savefig(DIR_fig+"figs5.svg")
# 
# 
# 
# 
left=0.07
right=0.0029
top=0.16
bottom=0.27


figure_S_5(left,right,top,bottom)
# 
# 
# 
# 
def figure_S_6(left,right,top,bottom):
    fix_x = 7.08#unit in inch
    ws_x = 0.34#of the axis x lenght
    ws_y = 0.7
    distx = 0.48 #of the axis x lenght
    disty = 0.09
    ax_x = fix_x/(3+2*ws_x)
    ax_y = 0.75 * ax_x
    
    fix_y = (4. + 3*disty) * ax_y
    
    gs1 = GridSpec(1,3, bottom=bottom+(3+3*disty)*ax_y/fix_y, top=-top+1, left=left+0, right=-right+1,wspace=ws_x,hspace = ws_y)
    gs2 = GridSpec(1,3, bottom=bottom+(2+2*disty)*ax_y/fix_y, top=-top+(3+2*disty)*ax_y/fix_y, left=left+0, right=-right+1,wspace=ws_x,hspace = ws_y)
    gs3 = GridSpec(1,3, bottom=bottom+(1+disty)*ax_y/fix_y, top=-top+(2.+disty)*ax_y/fix_y, left=left+0, right=-right+1,wspace=ws_x,hspace = ws_y)
    gs4 = GridSpec(1,3, bottom=bottom+0, top=-top+(1)*ax_y/fix_y, left=left+0, right=-right+1,wspace=ws_x,hspace = ws_y)
    
    fig = plt.figure(figsize=(fix_x, fix_y))
    
    
    axA = []
    axB = []
    axC = []
    axD = []
    for i in range(3):
        axA.append(fig.add_subplot(gs1[i]))
        axB.append(fig.add_subplot(gs2[i]))
        axC.append(fig.add_subplot(gs3[i]))
        axD.append(fig.add_subplot(gs4[i]))
    ax_com = [axA,axB,axC,axD]
    
    
    
    
    
    
    
    
    
    
    
    bimodal_limit = -0.014, 0.265
    power_limit = 0,0.715


    for i,state in enumerate(["1_1","4_2","4_4","4_6"]):
        ax_com[i][0].plot(data["prestimulus_spontaneous_analysis"][state]["2_column"]["instance"][::10],color=colors[state],alpha=0.8)


        ax_com[i][0].set_xlabel("Time (s)")
        ax_com[i][0].set_xticks([0,2000,4000])
        ax_com[i][0].set_xticklabels([0,2,4])
        ax_com[i][0].set_ylim(0,30)
        # ax_com[i][0].set_ylabel(r"$\beta_{\mathrm{intra}}=$"+"{},  ".format(state.split("_")[0])+r"$\beta_{\mathrm{inter}}=$"+"{}".format(state.split("_")[1])+"\n\nFiring Rate (Hz)")
        

        ylimmin,ylimmax=np.inf,-np.inf
        
        hist = data["prestimulus_spontaneous_analysis"][state]["2_column"]["hist"]/100
        hist_std = data["prestimulus_spontaneous_analysis"][state]["2_column"]["hist_std"]/100
        hist_bin = data["prestimulus_spontaneous_analysis"][state]["2_column"]["hist_bin"]

        ax_com[i][1].plot(hist_bin,hist,color=colors[state],alpha=0.8)
        ax_com[i][1].fill_between(hist_bin,hist-hist_std,hist+hist_std,color=colors[state],alpha=0.2)

        ylim = ax_com[i][1].get_ylim()
        ylimmin = min(ylimmin,ylim[0])
        ylimmax = max(ylimmax,ylim[1])
        ax_com[i][1].set_xlabel("Firing Rate (Hz)")


    ax_com[0][0].set_ylabel("NREM"+"\n\nFiring Rate (Hz)")
    ax_com[1][0].set_ylabel("W ("+r"$\bf{LS}$"+")"+"\n\nFiring Rate (Hz)")
    ax_com[2][0].set_ylabel("W ("+r"$\bf{H}$"+")"+"\n\nFiring Rate (Hz)")
    ax_com[3][0].set_ylabel("W ("+r"$\bf{DS}$"+")"+"\n\nFiring Rate (Hz)")
        
    for i in range(4):
        ax_com[i][1].set_ylim(bimodal_limit)



    axinylimmin,axinylimmax=np.inf,-np.inf
    ylimmin,ylimmax=np.inf,-np.inf
    for i,state in enumerate(["1_1","4_2","4_4","4_6"]):
        rel_pow = data["prestimulus_spontaneous_analysis"][state]["2_column"]["rel_pow"]
        rel_pow_std = data["prestimulus_spontaneous_analysis"][state]["2_column"]["rel_pow_std"]
        ax_com[i][2].bar(np.arange(len(rel_pow)), rel_pow, yerr=rel_pow_std, align='center', alpha=0.8,color=colors[state], ecolor='black', capsize=2, error_kw={"elinewidth":0.8,"capthick": 0.8})
        ax_com[i][2].set_xticks(np.arange(len(rel_pow)))
        ax_com[i][2].set_xticklabels(["SO","Delta","Theta","Alpha","Beta","Gamma"],rotation = 45)

        ylim = ax_com[i][2].get_ylim()
        ylimmin = min(ylimmin,ylim[0])
        ylimmax = max(ylimmax,ylim[1])


        axinylimmin = min(axinylimmin,rel_pow[-1]-rel_pow_std[-1])
        axinylimmax = max(axinylimmax,rel_pow[-1]+rel_pow_std[-1])


    for i in range(4):
        ax_com[i][2].set_ylim(power_limit) 

        ax_com[i][1].set_ylabel("Occurrence")

        ax_com[i][2].set_ylabel("Power")

    axA[0].set_title("One Representative Trial")
    axA[1].set_title("Histogram")
    axA[2].set_title("Power Spectrum")
    
    
    
    
    
    
    axA[0].text(-0.325,1.15,r"$\bf{a}$"+"(i)",horizontalalignment='left', transform=axA[0].transAxes, fontsize=7)
    axB[0].text(-0.325,1.15,r"$\bf{b}$"+"(i)",horizontalalignment='left', transform=axB[0].transAxes, fontsize=7)
    axC[0].text(-0.325,1.15,r"$\bf{c}$"+"(i)",horizontalalignment='left', transform=axC[0].transAxes, fontsize=7)
    axD[0].text(-0.325,1.15,r"$\bf{d}$"+"(i)",horizontalalignment='left', transform=axD[0].transAxes, fontsize=7)
    
    for j in range(4):
        ax_com[j][1].text(-0.23,1.15,"(ii)",horizontalalignment='left', transform=ax_com[j][1].transAxes, fontsize=7)
        ax_com[j][2].text(-0.23,1.15,"(iii)",horizontalalignment='left', transform=ax_com[j][2].transAxes, fontsize=7)
    
    
    
    fig.savefig(DIR_fig+"figs6.svg")
# 
# 
# 
# 
left=0.084
right=0.002
top=0.036
bottom=0.062


figure_S_6(left,right,top,bottom)
# 
# 
# 
# 
def figure_S_7(left,right,top,bottom):
    fix_x = 7.08#unit in inch
    ws_x = 0.35 #of the axis x lenght
    ws_y = 0.8
    distx = 0.02 #of the axis x lenght
    disty = 0.75
    ax_x = fix_x/(3+2*distx)
    ax_y = 0.5 * ax_x
    
    fix_y = (2 + 1*ws_y) * ax_y
    
    gs1 = GridSpec(2,1, bottom=bottom, top=-top+1, left=left, right=-right+(1)*ax_x/fix_x, wspace=ws_x,hspace = ws_y)
    gs2 = GridSpec(2,1, bottom=bottom, top=-top+1, left=left+(1+distx)*ax_x/fix_x, right=-right+(2+distx)*ax_x/fix_x, wspace=ws_x,hspace = ws_y)
    gs3 = GridSpec(2,1, bottom=bottom, top=-top+1, left=left+(2+2*distx)*ax_x/fix_x, right=-right+1, wspace=ws_x,hspace = ws_y)
    fig = plt.figure(figsize=(fix_x, fix_y))
    
    
    
    
    ax1 = []
    for i in range(2):
        ax1.append(fig.add_subplot(gs1[i]))
    ax2 = []
    for i in range(2):
        ax2.append(fig.add_subplot(gs2[i]))
    ax3 = []
    for i in range(2):
        ax3.append(fig.add_subplot(gs3[i]))
    
    
    
    
    
    
    
    
    
    
    
    n_column = 2
    
    pop = "pyr"
    
    
    
    
    
    for ic,column in enumerate(columns):
    
        bintra,binter = 1,1
        state = "{}_{}".format(bintra,binter)
        y = np.zeros(len(INPUTS),dtype=float)
        for i,inpu in enumerate(INPUTS):
            y[i] = data['deterministic_response'][state]["{}_column".format(n_column)]["input:{}".format(inpu)]["{}".format(column)]["{}".format(pop)]["firing"][0,11000]
        y = y - data['deterministic_response'][state]["{}_column".format(n_column)]["input:{}".format(inpu)]["{}".format(column)]["{}".format(pop)]["firing"][0,9000]
        ax1[ic].plot(INPUTS,y, 'o',color=colors[state],alpha=0.8,markersize=markersize)
        line1 = Line2D([], [], color="white", marker='o', markerfacecolor=colors["1_1"],alpha=0.8,markersize=markersize)
        
        
        
        
        line2 = []
        
    
    
        binput = 2
        for i,bintra in enumerate([2,6]):
            state = "{}_{}".format(bintra,binput)
            y = np.zeros(len(INPUTS),dtype=float)
            for i,inpu in enumerate(INPUTS):
                y[i] = data['deterministic_response'][state]["{}_column".format(n_column)]["input:{}".format(inpu)]["{}".format(column)]["{}".format(pop)]["firing"][0,11000]
            y = y - data['deterministic_response'][state]["{}_column".format(n_column)]["input:{}".format(inpu)]["{}".format(column)]["{}".format(pop)]["firing"][0,9000]
            ax1[ic].plot(INPUTS,y, 'o',color=colors[state],alpha=0.8,markersize=markersize)
            
        bintra = 2
        for i,binput in enumerate([6]):
            state = "{}_{}".format(bintra,binput)
            y = np.zeros(len(INPUTS),dtype=float)
            for i,inpu in enumerate(INPUTS):
                y[i] = data['deterministic_response'][state]["{}_column".format(n_column)]["input:{}".format(inpu)]["{}".format(column)]["{}".format(pop)]["firing"][0,11000]
            y = y - data['deterministic_response'][state]["{}_column".format(n_column)]["input:{}".format(inpu)]["{}".format(column)]["{}".format(pop)]["firing"][0,9000]
            ax1[ic].plot(INPUTS,y, 'o',color=colors[state],alpha=0.8,markersize=markersize)
        # l = ax1[ic].legend([line1], ["NREM"],loc=2)
        
        
        ax1[ic].set_xticks([0.01,0.05,0.09])
        ax1[ic].set_xticklabels([10,50,90])
            
        ax1[ic].set_xlabel("Stimulus Intensity (Hz)") 
    
        
        
        
    ax1[0].set_ylabel("Amplitude (Hz)")
    ax1[1].set_ylabel("Amplitude (Hz)")
    ax1[0].set_yticks([0.,3,6])
    ax1[1].set_yticks([0.,0.6,1.2])
    
    ylimmin = ax1[1].get_ylim()[0]
    ylimmax = ax1[0].get_ylim()[1]
    
    ax1[0].set_ylim(ylimmin,ylimmax)
    
    
    
    column="pert"
    pop="pyr"
    inpu = 0.05
    
    
    state = "2_2"
    y1 = data['deterministic_response'][state]["2_column"]["input:{}".format(inpu)][column]["pyr"]["firing"][0,11000]
    y1 = y1 - data['deterministic_response'][state]["2_column"]["input:{}".format(inpu)][column]["pyr"]["firing"][0,9000]
    y1 = 0.97*y1
    state = "6_2"
    y2 = data['deterministic_response'][state]["2_column"]["input:{}".format(inpu)][column]["pyr"]["firing"][0,11000]
    y2 = y2 - data['deterministic_response'][state]["2_column"]["input:{}".format(inpu)][column]["pyr"]["firing"][0,9000]
    y2 = 1.15*y2
    
    ax1[0].arrow(0.055,y1,0,y2-y1,width=0.00001, head_width=0.002, head_length=0.2625, fc='k', ec='k')
    ax1[0].text(0.6, 0.21, "pulling", transform=ax1[0].transAxes,fontsize=7,rotation=90)
    
    
    
    state = "2_2"
    y1 = data['deterministic_response'][state]["2_column"]["input:{}".format(inpu)][column]["pyr"]["firing"][0,11000]
    y1 = y1 - data['deterministic_response'][state]["2_column"]["input:{}".format(inpu)][column]["pyr"]["firing"][0,9000]
    y1 = 1.03*y1
    state = "2_6"
    y2 = data['deterministic_response'][state]["2_column"]["input:{}".format(inpu)][column]["pyr"]["firing"][0,11000]
    y2 = y2 - data['deterministic_response'][state]["2_column"]["input:{}".format(inpu)][column]["pyr"]["firing"][0,9000]
    y2 = 0.97*y2
    
    ax1[0].arrow(0.055,y1,0,y2-y1,width=0.00001, head_width=0.002, head_length=0.2625, fc='k', ec='k')
    
    ax1[0].text(0.6, 0.57, "driving", transform=ax1[0].transAxes,fontsize=7,rotation=90)
    
    
    
    
    
    column="unpert"
    pop="pyr"
    inpu = 0.05
    
    
    state = "2_2"
    y1 = data['deterministic_response'][state]["2_column"]["input:{}".format(inpu)][column]["pyr"]["firing"][0,11000]
    y1 = y1 - data['deterministic_response'][state]["2_column"]["input:{}".format(inpu)][column]["pyr"]["firing"][0,9000]
    y1 = 0.95*y1
    state = "6_2"
    y2 = data['deterministic_response'][state]["2_column"]["input:{}".format(inpu)][column]["pyr"]["firing"][0,11000]
    y2 = y2 - data['deterministic_response'][state]["2_column"]["input:{}".format(inpu)][column]["pyr"]["firing"][0,9000]
    y2 = 2*y2
    
    ax1[1].arrow(0.055,y1,0,y2-y1,width=0.00001, head_width=0.002, head_length=0.06, fc='k', ec='k')
    ax1[1].text(0.6, 0.06, "pulling", transform=ax1[1].transAxes,fontsize=7,rotation=90)
    
    
    
    state = "2_2"
    y1 = data['deterministic_response'][state]["2_column"]["input:{}".format(inpu)][column]["pyr"]["firing"][0,11000]
    y1 = y1 - data['deterministic_response'][state]["2_column"]["input:{}".format(inpu)][column]["pyr"]["firing"][0,9000]
    y1 = 1.05*y1
    state = "2_6"
    y2 = data['deterministic_response'][state]["2_column"]["input:{}".format(inpu)][column]["pyr"]["firing"][0,11000]
    y2 = y2 - data['deterministic_response'][state]["2_column"]["input:{}".format(inpu)][column]["pyr"]["firing"][0,9000]
    y2 = 0.95*y2
    
    
    ax1[1].arrow(0.055,y1,0,y2-y1,width=0.00001, head_width=0.002, head_length=0.06, fc='k', ec='k')
    
    ax1[1].text(0.6, 0.45, "driving", transform=ax1[1].transAxes,fontsize=7,rotation=90)
    
    
    
    
    
    
    ax1[0].set_title("Perturbed Column")
    ax1[1].set_title("Unperturbed Column")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    inpu = 0.05
    for ij,column in enumerate(["pert","unpert"]):
        
        state = "{}_{}".format(1,1)
        
        y = data['deterministic_response'][state]["2_column"]["input:{}".format(inpu)][column]["pyr"]["firing"][0,11000]
        y = y - data['deterministic_response'][state]["2_column"]["input:{}".format(inpu)][column]["pyr"]["firing"][0,9000]
        
        
        ax2[ij].plot(0,y,"o",color=colors[state],alpha=0.8,markersize=markersize)
        for k in range(len(STATES_ORG)):
            for i,state in enumerate(STATES_ORG[k]):
                y = data['deterministic_response'][state]["2_column"]["input:{}".format(inpu)][column]["pyr"]["firing"][0,11000]
                y = y - data['deterministic_response'][state]["2_column"]["input:{}".format(inpu)][column]["pyr"]["firing"][0,9000]
                ax2[ij].plot(1+k+0.15 * (i-1),y, "o",color=colors[state],alpha=0.8,markersize=markersize)
        ax2[ij].set_ylabel("Amplitude (Hz)")
        ax2[ij].set_xticks([0,1,2,3])
        ax2[ij].set_xticklabels(["NREM", "W ("+r"$\bf{LS}$"+")", "W ("+r"$\bf{H}$"+")", "W ("+r"$\bf{DS}$"+")"],rotation=30)
        
        
        ax2[ij].text(0.24, 0.87, "Stimulus={} Hz".format(int(1000*inpu)), transform=ax2[ij].transAxes,fontsize=7)
        
        
    ax2[0].set_title("Perturbed Column")
    ax2[1].set_title("Unperturbed Column")  
    
    for i in range(2):
        ylimmin,ylimmax = np.inf,-np.inf
        ylim = ax1[i].get_ylim()
        ylimmin = min(ylimmin,ylim[0])
        ylimmax = max(ylimmax,ylim[1])
        ax2[i].set_ylim(ylimmin,ylimmax)
    ax2[1].set_yticks([0,0.6,1.2])
        
        
        
    
    
    
    
    
    ei = np.zeros((3,3),dtype=float)
    for i,bintra in enumerate([2,4,6]):
        for j,binter in enumerate([2,4,6]):
            ei_aux=data["deterministic_response"]["{}_{}".format(bintra,binter)]["2_column"]["input:0.05"]["unpert"]["pyr"]['E/I_post']
            ei_aux = ei_aux[0]+ei_aux[1]+ei_aux[3]-ei_aux[2]
            ei[j,i] = ei_aux
    
    g1 = sns.heatmap(ei, cmap = 'gray_r',center=0.998*(ei.max() + ei.min())/2, cbar_kws={'label': r'$|\mathrm{E}|-|\mathrm{I}|$'+" (mV)",'aspect': 10},ax=ax3[0])
    g1.invert_yaxis()
    g1.set_xlabel(r"$\beta_{\mathrm{intra}}$")
    g1.set_ylabel(r"$\beta_{\mathrm{inter}}$")
    g1.set_yticks([0.5,1.5,2.5])
    g1.set_yticklabels([2,4,6])
    
    g1.set_xticks([0.5,1.5,2.5])
    g1.set_xticklabels([2,4,6])
        
    
    
    for i,bintra in enumerate([2,4,6]):
        for j,binter in enumerate([2,4,6]):
            ei_aux=data["deterministic_response"]["{}_{}".format(bintra,binter)]["2_column"]["input:0.05"]["unpert"]["pyr"]['E/I_post']
            ei_aux = ei_aux[0]+ei_aux[1]+ei_aux[3]-ei_aux[2]
    
    
            ei_aux_1=data["deterministic_response"]["{}_{}".format(bintra,binter)]["2_column"]["input:0.05"]["unpert"]["pyr"]['firing']
            ei_aux_1 = ei_aux_1[0,11000] - ei_aux_1[0,9000]
    
    
            state = "{}_{}".format(bintra,binter)
            ax3[1].plot(ei_aux,ei_aux_1,"o",c=colors[state],alpha=0.8,markersize=markersize)
    
    ax3[1].text(0.24, 0.87, "Stimulus=50 Hz", transform=ax3[1].transAxes,fontsize=7)
    ax3[1].set_xlabel(r'$|\mathrm{E}|-|\mathrm{I}|$'+" (mV)")
    ax3[1].set_ylabel("Amplitude (Hz)")
    
    
    ax3[0].set_title("Unperturbed Column") 
    ax3[1].set_title("Unperturbed Column") 
    
    ylimmin,ylimmax = np.inf,-np.inf
    ylim = ax1[1].get_ylim()
    ylimmin = min(ylimmin,ylim[0])
    ylimmax = max(ylimmax,ylim[1])
    ax3[1].set_ylim(ylimmin,ylimmax)
    ax3[1].set_yticks([0,0.6,1.2])
    
    
        
    fig.align_ylabels(ax1[:])
    fig.align_ylabels(ax2[:])
    fig.align_ylabels(ax3[:])
    
    ax1[0].text(-0.26,1.15,r"$\bf{a}$"+"(i)",horizontalalignment='left', transform=ax1[0].transAxes, fontsize=7)
    ax1[1].text(-0.245,1.15,"(ii)",horizontalalignment='left', transform=ax1[1].transAxes, fontsize=7)
    
    ax2[0].text(-0.26,1.15,r"$\bf{b}$"+"(i)",horizontalalignment='left', transform=ax2[0].transAxes, fontsize=7)
    ax2[1].text(-0.245,1.15,"(ii)",horizontalalignment='left', transform=ax2[1].transAxes, fontsize=7)
    
    ax3[0].text(-0.34,1.15,r"$\bf{c}$"+"(i)",horizontalalignment='left', transform=ax3[0].transAxes, fontsize=7)
    ax3[1].text(-0.245,1.15,"(ii)",horizontalalignment='left', transform=ax3[1].transAxes, fontsize=7)
    
    
    fig.savefig(DIR_fig+"figs7.svg")
# 
# 
# 
# 
left=0.06
right=0.05
top=0.079
bottom=0.11


figure_S_7(left,right,top,bottom)
# 
# 
# 
# 
def figure_S_8(left,right,top,bottom):   
    fix_x = 7.08#unit in inch
    ws_x = 0.15 #of the axis x lenght
    ws_y = 0.85
    distx = 0.48 #of the axis x lenght
    disty = 0.8
    ax_x = fix_x/(4+3*ws_x)
    ax_y = 0.7 * ax_x
    
    fix_y = (2+ws_y) * ax_y
    
    
    gs3 = GridSpec(2,4, bottom=bottom+0, top=-top+1, left=left+0, right=-right+1,wspace=ws_x,hspace = ws_y)
    
    fig = plt.figure(figsize=(fix_x, fix_y))
    
    
    
    ax = []
    for j in range(2):
        for i in range(4):
            ax.append(fig.add_subplot(gs3[j,i]))
    
    ax = np.array(ax).reshape(2,4)
    
    
    n_column = 2
    column = "pert"
    pop = "pyr"
    method = "k_means"
    
    
    t_crit = -stats.t.ppf(0.05/2,10)
    
    
    for ic, column in enumerate(["pert","unpert"]):
        ylimmin,ylimmax = np.inf,-np.inf
        for ij,inpu in enumerate([0.01,0.03,0.07,0.09]):
            
            if inpu!=0.05:
                state = "{}_{}".format(1,1)
                y = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][0,ij]
                yerr = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][1,ij]
                yerr = t_crit * yerr / np.sqrt(10)
                yerr = yerr
                ax[ic,ij].errorbar(0,y,yerr=yerr, fmt="o",color=colors[state],capsize=capsize, markersize=markersize, capthick=capthick, lw=lw, markeredgewidth=markeredgewidth,alpha=0.8)
                for k in range(len(STATES_ORG)):
                    for i,state in enumerate(STATES_ORG[k]):
                        y = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][0,ij]
                        yerr = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][1,ij]
                        yerr = t_crit * yerr / np.sqrt(10)
                        ax[ic,ij].errorbar(1+k+0.15 * (i-1),y,yerr=yerr, fmt="o",color=colors[state],capsize=capsize, markersize=markersize, capthick=capthick, lw=lw, markeredgewidth=markeredgewidth,alpha=0.8)
                
                ax[ic,ij].set_xticks([0,1,2,3])
                ax[ic,ij].set_xticklabels(["NREM", "W ("+r"$\bf{LS}$"+")", "W ("+r"$\bf{H}$"+")", "W ("+r"$\bf{DS}$"+")"],rotation=30)
                # ax[ic,ij].set_xticklabels(['\n N',
                #                          r"$\beta_{\mathrm{intra}}>\beta_{\mathrm{inter}}>1$", 
                #                          r"$\beta_{\mathrm{intra}}=\beta_{\mathrm{inter}}>1$", 
                #                          r"$\beta_{\mathrm{inter}}>\beta_{\mathrm{intra}}>1$"], rotation=30)
                # ax[ic,ij].set_title("Information Detection")
                ylim = ax[ic,ij].get_ylim()
                ylimmin = min(ylimmin,ylim[0])
                ylimmax = max(ylimmax,ylim[1])
                
                ax[ic,ij].text(0.04, 0.87, "Stimulus={} Hz".format(int(1000*inpu)), transform=ax[ic,ij].transAxes,fontsize=7)
        
        for i in range(4):
            ax[ic,i].set_ylim(ylimmin,ylimmax)
    for i in range(4):
        ax[0,i].set_yticks([0,0.5,1])
        ax[0,i].set_yticklabels([])
    ax[0,0].set_yticklabels([0,0.5,1])
    ax[0,0].set_ylabel("Information\nDetection (NMI)")


    for i in range(4):
        ax[1,i].set_yticks([0,0.1,0.2])
        ax[1,i].set_yticklabels([])
    ax[1,0].set_yticklabels([0,0.1,0.2])
    ax[1,0].set_ylabel("Information\nDetection (NMI)")
    
    
    
    ax[0,0].text(-0.32,1.15,r"$\bf{a}$"+"(i)",horizontalalignment='left', transform=ax[0,0].transAxes, fontsize=7)
    ax[1,0].text(-0.32,1.15,r"$\bf{b}$"+"(i)",horizontalalignment='left', transform=ax[1,0].transAxes, fontsize=7)

    for ic in range(2):
        for i,ind in enumerate(["(ii)","(iii)","(iiii)"]):
            ax[ic,i+1].text(-0.07,1.15,ind,horizontalalignment='left', transform=ax[ic,i+1].transAxes, fontsize=7)
    
    for i in range(4):
        ax[0,i].set_title("Perturbed Column") 
        ax[1,i].set_title("Unperturbed Column") 
    
    
    
    fig.savefig(DIR_fig+"figs8.svg")
# 
# 
# 
# 
left=0.07
right=0.005
top=0.08
bottom=0.11


figure_S_8(left,right,top,bottom)
# 
# 
# 
# 
def figure_S_9(left,right,top,bottom):
    fix_x = 7.08#unit in inch
    ws_x = 0.4 #of the axis x lenght
    ws_y = 0.6
    distx = 0.45 #of the axis x lenght
    disty = 1.8
    ax_x = fix_x/(3+2*ws_x)
    ax_y = 0.7 * ax_x
    
    fix_y = (2 + 1* ws_y) * ax_y
    
    
    
    gs1 = GridSpec(2,3, bottom=bottom, top=-top+1, left=left, right=-right+1, wspace=ws_x,hspace = ws_y)
    
    fig = plt.figure(figsize=(fix_x, fix_y))
    
    
    axA = []
    for i in range(2):
        for j in range(3):
            axA.append(fig.add_subplot(gs1[i,j]))
    
    
    axA = np.array(axA).reshape(2,3)
    
    t_crit = -stats.t.ppf(0.05/2,10)
    
    
    n_column = 2
    column = "pert"
    pop = "pyr"
    
    method = "logistic"
    
        
    state = "{}_{}".format(1,1)
    y = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][0,2]
    yerr = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][1,2]
    yerr = t_crit * yerr / np.sqrt(10)
    yerr = yerr
    axA[0,0].errorbar(0,y,yerr=yerr, fmt="o",color=colors[state],capsize=capsize, markersize=markersize, capthick=capthick, lw=lw, markeredgewidth=markeredgewidth,alpha=0.8)
    for k in range(len(STATES_ORG)):
        for i,state in enumerate(STATES_ORG[k]):
            y = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][0,2]
            yerr = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["detec_{}".format(method)][0][1,2]
            yerr = t_crit * yerr / np.sqrt(10)
            axA[0,0].errorbar(1+k+0.2 * (i-1),y,yerr=yerr, fmt="o",color=colors[state],capsize=capsize, markersize=markersize, capthick=capthick, lw=lw, markeredgewidth=markeredgewidth,alpha=0.8)
    axA[0,0].set_ylabel("Information\nDetection (NMI)")
    axA[0,0].set_xticks([0,1,2,3])
    axA[0,0].set_xticklabels(["NREM", "W ("+r"$\bf{LS}$"+")", "W ("+r"$\bf{H}$"+")", "W ("+r"$\bf{DS}$"+")"])
    axA[0,0].text(0.04, 0.87, "Stimulus=50 Hz", transform=axA[0,0].transAxes,fontsize=7)
    
    
    
    state = "{}_{}".format(1,1)
    y = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["diff_{}".format(method)][0][0]
    yerr = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["diff_{}".format(method)][0][1]
    yerr = t_crit * yerr / np.sqrt(10)
    yerr = yerr
    axA[1,0].errorbar(0,y,yerr=yerr, fmt="o",color=colors[state],capsize=capsize, markersize=markersize, capthick=capthick, lw=lw, markeredgewidth=markeredgewidth,alpha=0.8)
    for k in range(len(STATES_ORG)):
        for i,state in enumerate(STATES_ORG[k]):
            y= data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["diff_{}".format(method)][0][0]
            yerr = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["diff_{}".format(method)][0][1]
            yerr = t_crit * yerr / np.sqrt(10)
            axA[1,0].errorbar(1+k+0.2 * (i-1),y,yerr=yerr, fmt="o",color=colors[state],capsize=capsize, markersize=markersize, capthick=capthick, lw=lw, markeredgewidth=markeredgewidth,alpha=0.8)
    
    axA[1,0].set_xticks([0,1,2,3])
    axA[1,0].set_xticklabels(["NREM", "W ("+r"$\bf{LS}$"+")", "W ("+r"$\bf{H}$"+")", "W ("+r"$\bf{DS}$"+")"])
    
    
    
    
    
    # axA[0,0].set_yticks([0,0.4,0.8])
    
    
    # theoretical_score = 1/2
    # # axA[0,0].plot(INPUTS,[theoretical_score]*len(INPUTS),"--k")
    
    # theoretical_score = 1/len(INPUTS)
    # axA[1,0].plot([0,3],[theoretical_score]*2,"--k")
    axA[0,0].set_yticks([0.6,0.8,1])
    axA[1,0].set_yticks([0.28,0.64,1])
    
    
    
    
    
    state = "1_1"
    y = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["tvalue"][2]
    axA[0,1].plot(0,y, "o",color=colors[state],alpha=0.8,markersize=markersize)
    for k in range(len(STATES_ORG)):
        for i,state in enumerate(STATES_ORG[k]):
            y = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["tvalue"][2]
            axA[0,1].plot(1+k+0.2 * (i-1),y, "o",color=colors[state],alpha=0.8,markersize=markersize)
    
    
    axA[0,1].set_ylabel(r"$t$"+"-Value")
    axA[0,1].set_xticks([0,1,2,3])
    axA[0,1].set_xticklabels(["NREM", "W ("+r"$\bf{LS}$"+")", "W ("+r"$\bf{H}$"+")", "W ("+r"$\bf{DS}$"+")"])
    axA[0,1].text(0.04, 0.87, "Stimulus=50 Hz", transform=axA[0,1].transAxes,fontsize=7)
    
    
    
    
    
    
    state = "1_1"
    y = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["fvalue"]
    axA[1,1].plot(0,y,"o",color=colors[state],alpha=0.8, markersize=markersize)
    for k in range(len(STATES_ORG)):
        for i,state in enumerate(STATES_ORG[k]):
    
            y = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["fvalue"]
            axA[1,1].plot(1+k+0.2 * (i-1),y,"o",color=colors[state],alpha=0.8, markersize=markersize)
    axA[1,1].set_xticks([0,1,2,3])
    axA[1,1].set_yticks([0,2400,4800])
    axA[1,1].set_xticklabels(["NREM", "W ("+r"$\bf{LS}$"+")", "W ("+r"$\bf{H}$"+")", "W ("+r"$\bf{DS}$"+")"])
    axA[1,1].set_ylabel("F-Ratio")
    axA[1,1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # for state in ["1_1","6_2","4_4","2_6"]:
    #     y = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["mi_detect"]
    #     axA[0,3].plot(INPUTS,y, 'o',color=colors[state],alpha=0.8, markersize=markersize)
    
    
    
    # axA[0,3].set_ylabel("MI (Bits)")
    # axA[0,3].set_xticks([0.01,0.05,0.09])
    # axA[0,3].set_xlabel("Input Strength (ms"+r"$^{-1}$"+")")
    
    
    state = "1_1"
    y = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["mi"][0]
    axA[0,2].plot(0,y,"o",color=colors[state],alpha=0.8, markersize=markersize)
    for k in range(len(STATES_ORG)):
        for i,state in enumerate(STATES_ORG[k]):
    
            y = data["information_content"][state]["{}_column".format(n_column)]["{}".format(column)]["{}".format(pop)]["mi"][0]
            axA[0,2].plot(1+k+0.2 * (i-1),y,"o",color=colors[state],alpha=0.8, markersize=markersize)
    axA[0,2].set_xticks([0,1,2,3])
    axA[0,2].set_xticklabels(["NREM", "W ("+r"$\bf{LS}$"+")", "W ("+r"$\bf{H}$"+")", "W ("+r"$\bf{DS}$"+")"])
    axA[0,2].set_ylabel("MI (Bits)")
    
    
    
    
    
    
    
    
    # axA[0,0].set_ylabel("Information\nDetection\n\nMI (Bits)")
    # axA[1,0].set_ylabel("Information\nDifferentiation\n\nMI (Bits)")
    
    
    axA[0,0].set_ylabel("Information Detection\n(Accuracy)")
    axA[1,0].set_ylabel("Information Differentiation\n(Accuracy)")
    
    
    # axA[0,3].axis('off')
    
    
    
    
    
    
    
    
    # axA[0,0].set_title("K-means Clustering\n\n")
    axA[0,0].set_title("Logistic Regression")
    axA[0,1].set_title("Significance Test")
    axA[0,2].set_title("Information Theory")
    
    
    
    
    
    
    
    axA[0,0].text(-0.31,1.15,r"$\bf{a}$"+"(i)",horizontalalignment='left', transform=axA[0,0].transAxes, fontsize=7)
    axA[1,0].text(-0.31,1.15,"(ii)",horizontalalignment='left', transform=axA[1,0].transAxes, fontsize=7)
    
    axA[0,1].text(-0.22,1.15,r"$\bf{b}$"+"(i)",horizontalalignment='left', transform=axA[0,1].transAxes, fontsize=7)
    axA[1,1].text(-0.22,1.15,"(ii)",horizontalalignment='left', transform=axA[1,1].transAxes, fontsize=7)
    
    axA[0,2].text(-0.2,1.15,r"$\bf{c}$",horizontalalignment='left', transform=axA[0,2].transAxes, fontsize=7)
    
    
    fig.align_ylabels(axA[:, 0])
    fig.align_ylabels(axA[:, 1])
    
    axA[1,2].axis('off')
    
    fig.savefig(DIR_fig+"figs9.svg")
# 
# 
# 
# 
left=0.078
right=0.002
top=0.076
bottom=0.06


figure_S_9(left,right,top,bottom)
# 
# 
# 
# 
