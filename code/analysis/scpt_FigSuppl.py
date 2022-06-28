import fcn_megfmri
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt


# get custom colormaps
cmap_seq, cmap_seq_r, megcmap, megcmap2, categ_cmap = fcn_megfmri.make_colormaps()

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.sans-serif'] = ['Myriad Pro']
plt.rcParams['font.size'] = 18.0

# load schaefer info
schaeferpath = '../../data/schaefer/'
lhlabels = (schaeferpath + 'HCP/fslr32k/gifti/' +
            'Schaefer2018_400Parcels_7Networks_order_lh.label.gii')
rhlabels = (schaeferpath + 'HCP/fslr32k/gifti/' +
            'Schaefer2018_400Parcels_7Networks_order_rh.label.gii')

# path to ready-to-plot data
figData_path = '../../data/figures_data/'

####################################
# plot FigS1
####################################
# load band specific rsq data
rsq_band = pd.read_csv(figData_path + 'FigS1/FigS1_band_specific_rsq.csv')

# plot R-squared distribution
colors = sns.color_palette('hls', 7)
sns.displot(rsq_band, kind='kde', fill=True, height=8, aspect=0.7,
            palette=colors)
sns.despine(offset=5, trim=True)
plt.xlabel('R2')
plt.tight_layout()

# plot R-squared on brain surface
bands = list(rsq_band.columns)
for band in bands:
    toplot = rsq_band[band]
    brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                      vmin=np.percentile(toplot, 2.5),
                                      vmax=np.percentile(toplot, 97.5),
                                      colormap='viridis', customcmap=megcmap,
                                      colorbartitle=band,
                                      surf='inflated')

####################################
# plot FigS2
####################################
# load scfc coupling results
df_scfc = pd.read_csv(figData_path + 'FigS2/FigS2_scfc_coupling.csv')

scfc = np.array(df_scfc['scfc_coupling'])
rsq = np.array(df_scfc['rsq'])

# plot scfc coupling on the brain
toplot = np.array(scfc)
brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                  vmin=np.percentile(toplot, 0),
                                  vmax=np.percentile(toplot, 100),
                                  colormap='viridis', customcmap=megcmap,
                                  colorbartitle='scfc coupling',
                                  surf='inflated')

# correlate scfc coupling with R2 and plot
x = scfc
y = rsq

corr = stats.spearmanr(x, y)
pvalspin = fcn_megfmri.get_spinp(x, y, corrval=corr[0], nspin=10000,
                                 corrtype='spearman',
                                 lhannot=lhlabels, rhannot=rhlabels)

title = 'spearman r = %1.3f - p (spin) = %1.4f' % (corr[0], pvalspin)
xlab = 'sc-fc coupling (spearmanr)'
ylab = 'R-squared'
plt.figure()
fcn_megfmri.scatterregplot(x, y, title, xlab, ylab, 60)

####################################
# plot FigS3
####################################
### plot FigS3: panel A ###
# load ctf data for LCMV source reconstruction (original analysis)
df_ctf_lcmv = pd.read_csv(figData_path + 'FigS3/FigS3_A_ctf_lcmv.csv')

ctf_lcmv = np.array(df_ctf_lcmv['ctf_lcmv'])
rsq_lcmv = np.array(df_ctf_lcmv['rsq_lcmv (orig)'])

# plot source localization error for lcmv on the brain
toplot = np.array(ctf_lcmv)
brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                  vmin=np.percentile(toplot, 2.5),
                                  vmax=np.percentile(toplot, 97.5),
                                  colormap='viridis', customcmap=megcmap,
                                  colorbartitle='localization error LCMV',
                                  surf='inflated')

# correlate source localization error with R2 for lcmv and plot
x = ctf_lcmv
y = rsq_lcmv

corr = stats.spearmanr(x, y)
pvalspin = fcn_megfmri.get_spinp(x, y, corrval=corr[0], nspin=10000,
                                 corrtype='spearman',
                                 lhannot=lhlabels, rhannot=rhlabels)

title = 'spearman r = %1.3f - p (spin) = %1.4f' % (corr[0], pvalspin)
xlab = 'localization error LCMV'
ylab = 'R-squared LCMV (orig)'
plt.figure()
fcn_megfmri.scatterregplot(x, y, title, xlab, ylab, 60)


### plot FigS3: panel B ###
# load ctf data for sLoreta source reconstruction (original analysis)
df_ctf_sloreta = pd.read_csv(figData_path + 'FigS3/FigS3_B_ctf_sloreta.csv')

ctf_sloreta = np.array(df_ctf_sloreta['ctf_sloreta'])
rsq_sloreta = np.array(df_ctf_sloreta['rsq_sloreta'])

# plot source localization error for sloreta on the brain
toplot = np.array(ctf_sloreta)
brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                  vmin=np.percentile(toplot, 2.5),
                                  vmax=np.percentile(toplot, 97.5),
                                  colormap='viridis', customcmap=megcmap,
                                  colorbartitle='localization error sLoreta',
                                  surf='inflated')

# correlate source localization error with R2 for sloreta and plot
x = ctf_sloreta
y = rsq_sloreta

corr = stats.spearmanr(x, y)
pvalspin = fcn_megfmri.get_spinp(x, y, corrval=corr[0], nspin=10000,
                                 corrtype='spearman',
                                 lhannot=lhlabels, rhannot=rhlabels)

title = 'spearman r = %1.3f - p (spin) = %1.4f' % (corr[0], pvalspin)
xlab = 'localization error sLoreta'
ylab = 'R-squared sLoreta'
plt.figure()
fcn_megfmri.scatterregplot(x, y, title, xlab, ylab, 60)

####################################
# plot FigS4
####################################
# load SNR
df_snr = pd.read_csv(figData_path + 'FigS4/FigS4_snr.csv')

snr = np.array(df_snr['SNR'])
rsq = np.array(df_snr['rsq'])

# plot SNR on the brain
toplot = np.array(snr)
brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                  vmin=np.percentile(toplot, 0),
                                  vmax=np.percentile(toplot, 100),
                                  colormap='viridis', customcmap=megcmap,
                                  colorbartitle='signal-to-noise ratio (SNR)',
                                  surf='inflated')

# correlate SNR with R2 and plot
x = snr
y = rsq

corr = stats.spearmanr(x, y)
pvalspin = fcn_megfmri.get_spinp(x, y, corrval=corr[0], nspin=10000,
                                 corrtype='spearman',
                                 lhannot=lhlabels, rhannot=rhlabels)

title = 'spearman r = %1.3f - p (spin) = %1.4f' % (corr[0], pvalspin)
xlab = 'SNR (dB)'
ylab = 'R-squared'
plt.figure()
fcn_megfmri.scatterregplot(x, y, title, xlab, ylab, 60)

####################################
# plot FigS5
####################################
# load upper triangles of band limited MEG connectivity
df_meg_uppertri = pd.read_csv(figData_path +
                              'FigS5/FigS5_meg_connectivity_uppertriangle.csv')
meg_uppertriangle = np.array(df_meg_uppertri)
bands = list(df_meg_uppertri.columns)

# plot heatmap of correlations between band limited MEG connectivity
fig, ax = plt.subplots(1, 1)
ax = sns.heatmap(np.corrcoef(meg_uppertriangle.T), cmap=cmap_seq,
                 vmin=-0.1,
                 vmax=np.max(np.corrcoef(meg_uppertriangle.T)),
                 linewidth=0.5,
                 xticklabels=bands, yticklabels=bands, square=True)
ax.axes.set_title('band MEG fc correlation')
ax.figure.set_figwidth(7)
ax.figure.set_figheight(6)
plt.tight_layout()
