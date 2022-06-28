import fcn_megfmri
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# get custom colormaps
cmap_seq, cmap_seq_r, megcmap, megcmap2, categ_cmap = fcn_megfmri.make_colormaps()

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.sans-serif'] = ['Myriad Pro']
plt.rcParams['font.size'] = 18.0


# path to ready-to-plot data
figData_path = '../../data/figures_data/'

####################################
# plot Fig1: panel A
####################################
# load meg data
megdata = pd.ExcelFile(figData_path +
                       'Fig1/Fig1_A_B_groupFCmeg_aec_orth_schaefer400.xlsx')
bands = pd.read_csv(figData_path + 'Fig1/Fig1_A_bands_ordering.csv')
bands = list(bands['bandOrder'])

avgFCmeg = []
for band in bands:
    df_megfc = pd.read_excel(megdata, band,index_col=0)
    avgFCmeg.append(np.array(df_megfc))
avgFCmeg = np.array(avgFCmeg)

# load fmri data
avgFCmri = pd.read_csv(figData_path + 'Fig1/Fig1_A_B_groupFCmri_schaefer400.csv',
                       index_col=0)
avgFCmri = np.array(avgFCmri)

# plot MEG connectivity
fig = plt.figure()
for n, band in enumerate(bands):
    ax = plt.subplot(2, 3, n+1)
    np.fill_diagonal(avgFCmeg[n, :, :], 0)
    myplot = ax.imshow(avgFCmeg[n, :, :], cmap=cmap_seq_r, vmin=0) #, vmax=0.6)
    plt.title('MEG fc - frequency band: %s' % (band))

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(myplot, cax=cax)

    myplot.figure.set_figwidth(20)
    myplot.figure.set_figheight(10)

# plot fMRI connectivity
fig = plt.figure()
np.fill_diagonal(avgFCmri, 0)
plt.imshow(avgFCmri, cmap=cmap_seq_r, vmin=-0.2, vmax=0.8)
plt.title('fMRI fc')
plt.colorbar()

####################################
# plot Fig1: panel B
####################################
# load meg data
megdata = pd.ExcelFile(figData_path +
                       'Fig1/Fig1_A_B_groupFCmeg_aec_orth_schaefer400.xlsx')
bands = pd.read_csv(figData_path + 'Fig1/Fig1_A_bands_ordering.csv')
bands = list(bands['bandOrder'])

avgFCmeg = []
for band in bands:
    df_megfc = pd.read_excel(megdata, band,index_col=0)
    avgFCmeg.append(np.array(df_megfc))
avgFCmeg = np.array(avgFCmeg)

# load fmri data
avgFCmri = pd.read_csv(figData_path + 'Fig1/Fig1_A_B_groupFCmri_schaefer400.csv',
                       index_col=0)
avgFCmri = np.array(avgFCmri)

# correlate fMRI fc with MEG fc
mask = np.mask_indices(400, np.triu, 1)

allCorr = []
for n, band in enumerate(bands):
    x = avgFCmri[mask]
    y = avgFCmeg[n, :, :][mask]

    corr = stats.pearsonr(x, y)
    allCorr.append(corr[0])

fig, ax = plt.subplots(1, 1)
plt.bar(np.arange(6), allCorr, tick_label=bands, width=0.7,
        color=[153/255, 153/255, 153/255])

plt.title('Pearson r between MEG fc and fMRI fc')
plt.xlabel('frequency bands')
plt.ylabel('Pearson r')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

####################################
# plot Fig1: panel C
####################################
# load data: predicted fc from regional model
full_pred_fc = pd.read_csv(figData_path + 'Fig1/Fig1_C_predicted_fc_regional.csv',
                           index_col=0)
full_pred_fc = np.array(full_pred_fc)

# visualize predicted fc from regional model
# heatmap
plt.figure()
plt.imshow(full_pred_fc, vmin=0, vmax=0.55,
           cmap=cmap_seq_r)
plt.title('predicted fmri fc (regional)')
plt.colorbar()

# load data: upper triangles of predicted fc from regional model and empirical fc
df_uppertri = pd.read_csv(figData_path +
                          'Fig1/Fig1_C_predicted_fc_regional_uppertri.csv')

pred_fc_uppertri = np.array(df_uppertri['predictedFC_regional'])
empiric_fc_uppertri = np.array(df_uppertri['empiricalFC'])

# scatter plot
xlab = 'empirical fmri fc - upper tri'
ylab = 'predicted fmri fc - upper tri (regional)'
title = 'regional model: pearson r = %1.3f' % (stats.pearsonr(empiric_fc_uppertri,
                                               pred_fc_uppertri)[0])
plt.figure()
myplot = fcn_megfmri.scatterregplot(empiric_fc_uppertri,
                                    pred_fc_uppertri,
                                    title, xlab, ylab, 50)
myplot.figure.set_figwidth(7)
myplot.figure.set_figheight(7)

####################################
# plot Fig1: panel D
####################################
# load data: predicted fc from global model
full_pred_fc_g = pd.read_csv(figData_path + 'Fig1/Fig1_D_predicted_fc_global.csv',
                             index_col=0)
full_pred_fc_g = np.array(full_pred_fc_g)


# visualize predicted fc from global model
# heatmap
plt.figure()
plt.imshow(full_pred_fc_g, vmin=0, vmax=0.35,
           cmap=cmap_seq_r)
plt.title('predicted fmri fc (global)')
plt.colorbar()

# load data: upper triangles of predicted fc from global model and empirical fc
df_uppertri_g = pd.read_csv(figData_path +
                          'Fig1/Fig1_D_predicted_fc_global_uppertri.csv')

pred_fc_uppertri_g = np.array(df_uppertri_g['predictedFC_global'])
empiric_fc_uppertri = np.array(df_uppertri_g['empiricalFC'])

# scatter plot
xlab = 'empirical fmri fc - upper tri'
ylab = 'predicted fmri fc - upper tri (global)'
title = 'global model: pearson r = %1.3f' % (stats.pearsonr(empiric_fc_uppertri,
                                            pred_fc_uppertri_g)[0])
plt.figure()
myplot = fcn_megfmri.scatterregplot(empiric_fc_uppertri,
                                    pred_fc_uppertri_g,
                                    title, xlab, ylab, 50)
myplot.figure.set_figwidth(7)
myplot.figure.set_figheight(7)

####################################
# plot Fig1: panel E
####################################
# load data
# regional rsq
rsq = pd.read_csv(figData_path + 'Fig1/Fig1_E_rsq_regional.csv')
rsq = np.array(rsq)
# global rsq
rsq_g = np.loadtxt(figData_path + '/Fig1/Fig1_E_rsq_global.txt')

# plot histogram distribution
fig, ax = plt.subplots(1, 1)
plt.hist(rsq, density=False, rwidth=0.9,
         color=[153/255, 153/255, 153/255], label='regional fit')
plt.vlines([rsq_g], ymin=0, ymax=115, linewidth=3,
           color=[242/255, 111/255, 136/255], label='global fit')
plt.xlabel('R-squared')
plt.ylabel('count')
plt.title('meg-fmri mapping')
plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
