import mayavi
import neuromaps
import fcn_megfmri
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.stats import multitest
from neuromaps.parcellate import Parcellater
from sklearn.preprocessing import MinMaxScaler


# load data
megdata = np.load('../../data/groupFCmeg_aec_orth_schaefer400.npy.npz')
avgFCmeg = megdata['megfc']
bands = megdata['bands']

avgFCmri = np.load('../../data/groupFCmri_schaefer400.npy')

# load schaefer info
schaeferpath = '../../data/schaefer/'
lhlabels = (schaeferpath + 'HCP/fslr32k/gifti/' +
            'Schaefer2018_400Parcels_7Networks_order_lh.label.gii')
rhlabels = (schaeferpath + 'HCP/fslr32k/gifti/' +
            'Schaefer2018_400Parcels_7Networks_order_rh.label.gii')
labelinfo = np.loadtxt(schaeferpath + 'HCP/fslr32k/gifti/' +
                       'Schaefer2018_400Parcels_7Networks_order_info.txt',
                       dtype='str', delimiter='tab')
rsnlabels = []
for row in range(0, len(labelinfo), 2):
    rsnlabels.append(labelinfo[row].split('_')[2])

# load coordinates and estimate distance
coor = np.loadtxt(schaeferpath + '/Schaefer_400_centres.txt', dtype=str)
coor = coor[:, 1:].astype(float)
distance = sklearn.metrics.pairwise_distances(coor)

# get custom colormaps
cmap_seq, cmap_seq_r, megcmap, megcmap2, categ_cmap = fcn_megfmri.make_colormaps()

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.sans-serif'] = ['Myriad Pro']
plt.rcParams['font.size'] = 18.0

####################################
# linear regression
####################################
# regional model
rsq, full_pred_fc, corrVal = fcn_megfmri.regional_lreg(megdata=avgFCmeg,
                                                       fmridata=avgFCmri,
                                                       distance=distance,
                                                       correct_dist=False,
                                                       adjusted_rsq=True)

# global model
rsq_g, full_pred_fc_g, corrVal_g = fcn_megfmri.global_lreg(megdata=avgFCmeg,
                                                           fmridata=avgFCmri,
                                                           distance=distance,
                                                           correct_dist=False,
                                                           adjusted_rsq=True)

####################################
# visualize predicted fc
####################################
nnode = len(rsq)
masku = np.mask_indices(nnode, np.triu, 1)

# regional model
plt.figure()
plt.imshow(full_pred_fc, vmin=0, vmax=0.55,
           cmap=cmap_seq_r)
plt.title('predicted fmri fc - upper tri (regional)')
plt.colorbar()

# scatter plot
xlab = 'empirical fmri fc'
ylab = 'predicted fmri fc'
title = 'regional model: pearson r = %1.3f' % (stats.pearsonr(avgFCmri[masku],
                                               full_pred_fc[masku])[0])
plt.figure()
myplot = fcn_megfmri.scatterregplot(avgFCmri[masku],
                                    full_pred_fc[masku],
                                    title, xlab, ylab, 50)
myplot.figure.set_figwidth(7)
myplot.figure.set_figheight(7)


# global model
plt.figure()
plt.imshow(full_pred_fc_g, vmin=0, vmax=0.35,
           cmap=cmap_seq_r)
plt.title('predicted fmri fc - upper tri (global)')
plt.colorbar()

# scatter plot
xlab = 'empirical fmri fc'
ylab = 'predicted fmri fc'
title = 'global model: pearson r = %1.3f' % (stats.pearsonr(avgFCmri[masku],
                                             full_pred_fc_g[masku])[0])
plt.figure()
myplot = fcn_megfmri.scatterregplot(avgFCmri[masku],
                                    full_pred_fc_g[masku],
                                    title, xlab, ylab, 50)
myplot.figure.set_figwidth(7)
myplot.figure.set_figheight(7)

####################################
# visualize R2 distribution
####################################
# plot regional R2 on brain
toplot = np.array(rsq)
brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                  vmin=np.percentile(toplot, 2.5),
                                  vmax=np.percentile(toplot, 97.5),
                                  colormap='viridis', customcmap=megcmap,
                                  colorbartitle='adjusted rsq',
                                  surf='inflated')


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

####################################
# functional hierarchy
####################################
# use fmri data from same subjects to get fc gradient
grads, lambdas = fcn_megfmri.get_gradients(avgFCmri, ncomp=10)

ncomp = 0
fcgrad = grads[:, ncomp]
toplot = fcgrad
brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                  vmin=np.percentile(toplot, 2.5),
                                  vmax=np.percentile(toplot, 97.5),
                                  colormap='viridis', customcmap=megcmap,
                                  colorbartitle='fc gradient %s' % (ncomp+1),
                                  surf='inflated')

# or use neuromaps to get fc gradient of all HCP subjects
grad1 = neuromaps.datasets.fetch_annotation(desc='fcgradient01')
parcellation = [lhlabels, rhlabels]
parc = Parcellater(parcellation, 'fslr')
parcellated_grad1 = parc.fit_transform(grad1, 'fslr')

# correlate and plot
x = fcgrad
y = rsq

corr = stats.spearmanr(x, y)
pvalspin = fcn_megfmri.get_spinp(x, y, corrval=corr[0], nspin=10000,
                                 corrtype='spearman',
                                 lhannot=lhlabels, rhannot=rhlabels)

title = 'spearman r = %1.3f - p (spin) = %1.4f' % (corr[0], pvalspin)
xlab = 'fc hierarchy'
ylab = 'R-squared'
plt.figure()
fcn_megfmri.scatterregplot(x, y, title, xlab, ylab, 60)

####################################
# intrinsic networks
####################################
# plot
network_labels = rsnlabels
rsnlabelsabb = np.unique(rsnlabels)
network_score = pd.DataFrame(data={'R-squared': rsq,
                             'rsn': network_labels})
medianVal = np.vstack([network_score.loc[network_score[
                      'rsn'].eq(netw),
                      'R-squared'].median() for netw in rsnlabelsabb])
idx = np.argsort(-medianVal.squeeze())
plot_order = [rsnlabelsabb[k] for k in idx]
sns.set(style='ticks', palette='pastel')
ax = sns.boxplot(x='rsn', y='R-squared',
                 data=network_score,
                 width=0.5, fliersize=3, showcaps=False,
                 order=plot_order, showfliers=False)

sns.despine(ax=ax, offset=5, trim=True)

plt.setp(ax.artists, edgecolor = 'k', facecolor='w')
plt.setp(ax.lines, color='k')

ax.figure.set_figwidth(7)
ax.figure.set_figheight(6)
plt.tight_layout()

####################################
# dominance analysis
####################################
# estimate band-specific contribution to R2 for each node
percentDominance_adj = fcn_megfmri.get_percent_dominance(megdata=avgFCmeg,
                                                         fmridata=avgFCmri,
                                                         distance=distance,
                                                         correct_dist=False,
                                                         adjusted_rsq=True)


# percent dominance boxplots
for n, band in enumerate(bands):
    if n == 0:
        bandContrib = pd.DataFrame(data={'percent': percentDominance_adj[:, n],
                                         'band': [band] * 400})
    else:
        temp = pd.DataFrame(data={'percent': percentDominance_adj[:, n],
                                  'band': [band] * 400})
        bandContrib = pd.concat([bandContrib, temp], axis=0)
bandContrib['percent'] = bandContrib['percent'].astype(float)
medianVal = np.vstack([bandContrib.loc[bandContrib[
                      'band'].eq(band),
                      'percent'].median() for band in bands])
idx = np.argsort(-medianVal.squeeze())
plot_order = [bands[k] for k in idx]
sns.set(style='ticks', palette='pastel')
plt.figure()
ax = sns.boxplot(x='band', y='percent', data=bandContrib,
                 width=.45, fliersize=3, showcaps=False,
                 order=plot_order, showfliers=False)

sns.despine(ax=ax, offset=5, trim=True)
ax.axes.set_title('Dominance analysis')

plt.setp(ax.artists, edgecolor = 'k', facecolor='w')
plt.setp(ax.lines, color='k')

ax.figure.set_figwidth(6)
ax.figure.set_figheight(6)
plt.tight_layout()

####################################
# percent dominance in intrinsic networks
####################################
percentDominancrsn = pd.DataFrame(percentDominance_adj, columns=bands)
percentDominancrsn['rsn'] = rsnlabels
bandrsncontrib = []
for band in bands:
    banddata = percentDominancrsn[band]
    tempdf = pd.DataFrame(banddata, columns=[band])
    tempdf['rsn'] = percentDominancrsn['rsn']
    meanrsnContrib = tempdf.groupby(['rsn']).mean()
    bandrsncontrib.append(meanrsnContrib)
orig_order = list(meanrsnContrib.index)
bandrsncontrib = np.array(bandrsncontrib).squeeze()
new_order = ['Default', 'Cont', 'Limbic', 'SalVentAttn', 'DorsAttn',
              'SomMot', 'Vis']
new_order_idx = [orig_order.index(rsnname) for rsnname in new_order]
reordered_bandrsncontrib = bandrsncontrib[:, new_order_idx]
ax = sns.heatmap(reordered_bandrsncontrib, cmap=cmap_seq, yticklabels=bands,
                 vmin=np.min(reordered_bandrsncontrib),
                 vmax=np.max(reordered_bandrsncontrib), linewidth=0.5,
                 xticklabels=new_order)
ax.axes.set_title('percent dominance for intrinsic nerworks')
ax.figure.set_figwidth(7)
ax.figure.set_figheight(6)
plt.tight_layout()

####################################
# maximum contributing band
####################################
# identify maximum contributing band for each node
nnode = len(percentDominance_adj)
maxContrib = []
for node in range(nnode):
    maxContrib.append(bands[np.argmax(percentDominance_adj[node, :])])

maxContribidx = np.zeros((nnode, 1))
for n, band in enumerate(bands):
    idx = np.where(np.array(maxContrib) == band)[0]
    maxContribidx[idx] = n

# plot on brain surface
toplot = maxContribidx
brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                  colormap='viridis', customcmap=categ_cmap,
                                  colorbartitle=('max contribution'),
                                  surf='inflated', vmin=0, vmax=5)

####################################
# band-specific contribution
####################################
# plot band contribution on surface
for n, band in enumerate(bands):
    toplot = percentDominance_adj[:, n]
    brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                      vmin=np.percentile(toplot, 2.5),
                                      vmax=np.percentile(toplot, 97.5),
                                      colormap='viridis', customcmap=megcmap2,
                                      colorbartitle=band,
                                      surf='inflated')

####################################
# linear regression: distance-dependant CV
####################################
# regional model with distance dependent cross-validation
train_corr, test_corr = fcn_megfmri.regional_lreg_cv(megdata=avgFCmeg,
                                                     fmridata=avgFCmri,
                                                     distance=distance,
                                                     coor=coor,
                                                     correct_dist=False,
                                                     train_pct=0.75,
                                                     verbose=True)

# plot on brain surface
toplot = test_corr
brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                  vmin=np.percentile(toplot, 2.5),
                                  vmax=np.percentile(toplot, 97.5),
                                  colormap='viridis', customcmap=megcmap,
                                  colorbartitle='training set - Pearson r',
                                  surf='inflated')

# correlate and plot
x = train_corr
y = test_corr

corr = stats.spearmanr(x, y)
pvalspin = fcn_megfmri.get_spinp(x, y, corrval=corr[0], nspin=10000,
                                 corrtype='spearman',
                                 lhannot=lhlabels, rhannot=rhlabels)

title = 'Spearman r = %1.3f - p (spin) = %1.4f' % (corr[0], pvalspin)
xlab = 'Pearson r - training set'
ylab = 'Pearson r - test set'
plt.figure()
fcn_megfmri.scatterregplot(x, y, title, xlab, ylab, 60)

####################################
# linear regression: subj-level (leave-one-subject out)
####################################
# subject data are not included
subjFCmri = np.load('../../data/subjFCmri_schaefer400.npy')
subjFCmeg = np.load('../../data/subjFCmeg_aec_orth_schaefer400.npy')

train_corr, test_corr = fcn_megfmri.regional_lreg_subj(subjmegdata=subjFCmeg,
                                                       subjfmridata=subjFCmri,
                                                       distance=distance,
                                                       correct_dist=False,
                                                       verbose=True)

# plot on brain surface
toplot = test_corr
brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                  vmin=np.percentile(toplot, 2.5),
                                  vmax=np.percentile(toplot, 97.5),
                                  colormap='viridis', customcmap=megcmap,
                                  colorbartitle='training set - Pearson r',
                                  surf='inflated')

# correlate and plot
x = train_corr
y = test_corr

corr = stats.spearmanr(x, y)
pvalspin = fcn_megfmri.get_spinp(x, y, corrval=corr[0], nspin=10000,
                                 corrtype='spearman',
                                 lhannot=lhlabels, rhannot=rhlabels)

title = 'Spearman r = %1.3f - p (spin) = %1.4f' % (corr[0], pvalspin)
xlab = 'Pearson r - training set'
ylab = 'Pearson r - test set'
plt.figure()
fcn_megfmri.scatterregplot(x, y, title, xlab, ylab, 60)

####################################
# ctf metric
####################################
ctf = np.load('../../data/avg_peak_err_ctf_resolutionMatrix_lcmv_Schaefer400.npy')

# correlate and plot
x = ctf
y = rsq

corr = stats.spearmanr(x, y)
pvalspin = fcn_megfmri.get_spinp(x, y, corrval=corr[0], nspin=10000,
                                corrtype='spearman',
                                lhannot=lhlabels, rhannot=rhlabels)

title = 'spearman r = %1.3f - p (spin) = %1.4f' % (corr[0], pvalspin)
xlab = 'localization error'
ylab = 'R-squared'
plt.figure()
fcn_megfmri.scatterregplot(x, y, title, xlab, ylab, 60)

# plot on brain surface
toplot = ctf
brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                  vmin=np.percentile(toplot, 2.5),
                                  vmax=np.percentile(toplot, 97.5),
                                  colormap='viridis', customcmap=megcmap,
                                  colorbartitle='localization error',
                                  surf='inflated')

####################################
# BigBrain profile intensity
####################################
profileIntensity = np.load('../../data/profileIntensity_schaefer400.npy')
scaler = MinMaxScaler()
profileIntensity_norm = 1-scaler.fit_transform(profileIntensity.T)

corr_rsq_intensity = np.zeros((1, 50))
pvalspin_rsq_intensity = np.zeros((1, 50))
for surf in range(50):
    x = stats.zscore(profileIntensity_norm[:, surf])
    y = rsq

    corr = stats.spearmanr(x, y)

    pvalspin = fcn_megfmri.get_spinp(x, y, corrval=corr[0], nspin=10000,
                                     corrtype='spearman',
                                     lhannot=lhlabels, rhannot=rhlabels)

    corr_rsq_intensity[0, surf] = corr[0]
    pvalspin_rsq_intensity[0, surf] = pvalspin

multicomp = multitest.multipletests(pvalspin_rsq_intensity.squeeze(),
                                    alpha=0.05, method='fdr_bh')

pointsize = np.zeros((50, 1))
pointsize = np.squeeze(pointsize)
pointsize[multicomp[1] > 0.05] = 1

fig, ax = plt.subplots(1, 1)
ax = sns.scatterplot(np.arange(50), corr_rsq_intensity.flatten(),
                     hue=corr_rsq_intensity.flatten(),
                     palette=cmap_seq, size=pointsize)

plt.xlabel('cortical depth')
plt.ylabel('Spearman rho - rsq vs profile intensity')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.figure.set_figwidth(8)
ax.figure.set_figheight(4)
plt.tight_layout()

# brain plots
layer = 0
toplot = profileIntensity_norm[:, layer]
brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                  vmin=np.percentile(toplot, 2.5),
                                  vmax=np.percentile(toplot, 97.5),
                                  colormap='viridis', customcmap=megcmap,
                                  colorbartitle=('Profile Intensity, layer %s'
                                                  % (layer)),
                                  surf='inflated')

####################################
# linear regression: band specific
####################################
# regional model
rsq, full_pred_fc, corrVal = fcn_megfmri.regional_lreg(megdata=avgFCmeg,
                                                        fmridata=avgFCmri,
                                                        distance=distance,
                                                        correct_dist=False,
                                                        adjusted_rsq=True)
rsq_band = pd.DataFrame(rsq, columns=['multiband'])
for n, band in enumerate(bands):
    rsq, _, _ = fcn_megfmri.regional_lreg(megdata=avgFCmeg[n,:,:][np.newaxis, :, :],
                                          fmridata=avgFCmri,
                                          distance=distance,
                                          correct_dist=False,
                                          adjusted_rsq=True)
    rsq_band[band] = rsq

# plot R-squared distribution
colors = sns.color_palette('hls', 7)
sns.displot(rsq_band, kind='kde', fill=True, height=8, aspect=0.7,
            palette=colors)
sns.despine(offset=5, trim=True)
plt.tight_layout()

# plot R-squared on brain surface
band = bands[n]
toplot = rsq_band[band]
brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                  vmin=np.percentile(toplot, 2.5),
                                  vmax=np.percentile(toplot, 97.5),
                                  colormap='viridis', customcmap=megcmap,
                                  colorbartitle=band,
                                  surf='inflated')
