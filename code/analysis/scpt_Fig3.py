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
# plot Fig3: panel A
####################################
# load data
# regional rsq
rsq = pd.read_csv(figData_path + '/Fig3/Fig3_A_rsq_regional.csv')

# band contribution from dominance analysis
bandContrib = pd.read_csv(figData_path +
                          'Fig3/Fig3_A_bandContribution_dominanceAnalysis.csv')
bands_ordering = pd.read_csv(figData_path + 'Fig3/Fig3_A_bands_ordering.csv')
bands = np.array(bands_ordering['bandOrder'])

# plot regional R2 on brain surface
toplot = np.array(rsq)
brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                  vmin=np.percentile(toplot, 2.5),
                                  vmax=np.percentile(toplot, 97.5),
                                  colormap='viridis', customcmap=megcmap,
                                  colorbartitle='adjusted rsq',
                                  surf='inflated')

# plot boxplot for band contributions
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
# plot Fig3: panel B
####################################
# load band contribution for intrinsic networks
df_bandrsncontrib = pd.read_csv(figData_path +
                                'Fig3/Fig3_B_bandContribution_rsn.csv')
bands = list(df_bandrsncontrib.columns[:-1])

bandrsncontrib = []
for band in bands:
    banddata = df_bandrsncontrib[band]
    tempdf = pd.DataFrame(banddata, columns=[band])
    tempdf['rsn'] = df_bandrsncontrib['rsn']
    meanrsnContrib = tempdf.groupby(['rsn']).mean()
    bandrsncontrib.append(meanrsnContrib)
orig_order = list(meanrsnContrib.index)
bandrsncontrib = np.array(bandrsncontrib).squeeze()
new_order = ['Default', 'Cont', 'Limbic', 'SalVentAttn', 'DorsAttn',
              'SomMot', 'Vis']
new_order_idx = [orig_order.index(rsnname) for rsnname in new_order]
reordered_bandrsncontrib = bandrsncontrib[:, new_order_idx]

# plot heatmap for band contribution in each intrinsic network
fig, ax = plt.subplots(1, 1)
ax = sns.heatmap(reordered_bandrsncontrib, cmap=cmap_seq, yticklabels=bands,
                 vmin=np.min(reordered_bandrsncontrib),
                 vmax=np.max(reordered_bandrsncontrib), linewidth=0.5,
                 xticklabels=new_order)
ax.axes.set_title('percent dominance for intrinsic nerworks')
ax.figure.set_figwidth(7)
ax.figure.set_figheight(6)
plt.tight_layout()

####################################
# plot Fig3: panel C
####################################
# load maximum contributing bands
df_maxContribBand = pd.read_csv(figData_path +
                                'Fig3/Fig3_C_maximumContributingBand.csv')
maxContribidx = np.array(df_maxContribBand['bandIdx'].astype(int))

# plot on brain surface
toplot = maxContribidx
brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                  colormap='viridis', customcmap=categ_cmap,
                                  colorbartitle=('max contribution'),
                                  surf='inflated', vmin=0, vmax=5)

####################################
# plot Fig3: panel D
####################################
# load percent dominace for all regions and all bands
df_percentDominance_adj = pd.read_csv(figData_path +
                                      'Fig3/Fig3_D_percentDominance.csv')
bands = list(df_percentDominance_adj.columns)
percentDominance_adj = np.array(df_percentDominance_adj)

# plot band contribution on surface
for n, band in enumerate(bands):
    toplot = percentDominance_adj[:, n]
    brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                      vmin=np.ceil(np.percentile(toplot, 2.5)),
                                      vmax=np.ceil(np.percentile(toplot, 97.5)),
                                      colormap='viridis', customcmap=megcmap2,
                                      colorbartitle=band,
                                      surf='inflated')
