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
labelinfo = np.loadtxt(schaeferpath + 'HCP/fslr32k/gifti/' +
                       'Schaefer2018_400Parcels_7Networks_order_info.txt',
                       dtype='str', delimiter='tab')
rsnlabels = []
for row in range(0, len(labelinfo), 2):
    rsnlabels.append(labelinfo[row].split('_')[2])

# path to ready-to-plot data
figData_path = '../../data/figures_data/'

####################################
# plot Fig2: panel A
####################################
# load data
# regional rsq
rsq = pd.read_csv(figData_path + '/Fig2/Fig2_A_rsq_regional.csv')

rsq = np.array(rsq['rsq'])


# plot histogram distribution
fig, ax = plt.subplots(1, 1)
plt.hist(rsq, density=False, rwidth=0.9,
         color=[153/255, 153/255, 153/255], label='regional fit')
plt.xlabel('R-squared')
plt.ylabel('count')
plt.title('meg-fmri mapping')
plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# plot regional R2 on brain surface
toplot = np.array(rsq)
brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                  vmin=np.percentile(toplot, 2.5),
                                  vmax=np.percentile(toplot, 97.5),
                                  colormap='viridis', customcmap=megcmap,
                                  colorbartitle='adjusted rsq',
                                  surf='inflated')

####################################
# plot Fig2: panel B
####################################
# load data
# first gradient of fMRI functional connectivity
df_fcgrad = pd.read_csv(figData_path + '/Fig2/Fig2_B_fcgrad1_rsq.csv')

fcgrad = np.array(df_fcgrad['fMRI FC Grad1'])
rsq = np.array(df_fcgrad['rsq'])

# correlate with regional R2 and plot
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

# plot fc gradient on brain surface
toplot = fcgrad
brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                  vmin=np.percentile(toplot, 2.5),
                                  vmax=np.percentile(toplot, 97.5),
                                  colormap='viridis', customcmap=megcmap,
                                  colorbartitle='fc gradient 1',
                                  surf='inflated')

####################################
# plot Fig2: panel C
####################################
# load data
network_score = pd.read_csv(figData_path + 'Fig2/Fig2_C_intrinsic_network_rsq.csv')

# plot boxplots
rsnlabelsabb = np.unique(network_score['rsn'])
medianVal = np.vstack([network_score.loc[network_score[
                      'rsn'].eq(netw),
                      'R-squared'].median() for netw in rsnlabelsabb])
idx = np.argsort(-medianVal.squeeze())
plot_order = [rsnlabelsabb[k] for k in idx]
sns.set(style='ticks', palette='pastel')

fig, ax = plt.subplots(1, 1)
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

# plot rsn on the brain
# get rsn label indices
uniqlabels, uniqidx = np.unique(rsnlabels, return_index=True)
uniqlabels = uniqlabels[np.argsort(uniqidx)]
rsnidx = np.zeros((400, 1))
for n, rsn in enumerate(uniqlabels):
    idx = np.where(np.array(rsnlabels) == rsn)[0]
    rsnidx[idx] = n

categ_cmap_rsn = fcn_megfmri.make_colormap_rsn()

# order is the same as the order in "uniqlabels"
toplot = rsnidx
brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                  colormap='viridis',
                                  customcmap=categ_cmap_rsn,
                                  colorbartitle=('Yeo rsn'),
                                  surf='inflated')

####################################
# plot Fig2: panel D
####################################
# load data
df_profileIntensity_norm = pd.read_csv(figData_path +
                                'Fig2/Fig2_D_normalized_intensity_profile.csv')
profileintensity_rsq_corr = pd.read_csv(figData_path +
                                'Fig2/Fig2_D_profileintensity_rsq_corr.csv')


# plot correlation values between BigBrain intensity profiles and regional R2
corr_value = np.array(profileintensity_rsq_corr['corrVal'])
fdr_pvalSpin = np.array(profileintensity_rsq_corr['fdr_pvalSpin'])

pointsize = np.zeros((50, 1))
pointsize = np.squeeze(pointsize)
pointsize[fdr_pvalSpin > 0.05] = 1

fig, ax = plt.subplots(1, 1)
ax = sns.scatterplot(np.arange(50), corr_value,
                     hue=corr_value,
                     palette=cmap_seq, size=pointsize, legend=None)

plt.xlabel('cortical depth')
plt.ylabel('Spearman rho - rsq vs profile intensity')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.figure.set_figwidth(8)
ax.figure.set_figheight(4)
plt.tight_layout()

# example brain plots for intensity profiles at first, middle, and last layers
profileIntensity_norm = np.array(df_profileIntensity_norm)
layers = [0, 24, 49]
for layer in layers:
    toplot = profileIntensity_norm[:, layer]
    brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                    vmin=np.percentile(toplot, 2.5),
                                    vmax=np.percentile(toplot, 97.5),
                                    colormap='viridis', customcmap=megcmap,
                                    colorbartitle=('Profile Intensity, layer %s'
                                                    % (layer+1)),
                                    surf='inflated')

####################################
# plot Fig2: panel E
####################################
# load NPY1R gene expression data
df_npy1r = pd.read_csv(figData_path + 'Fig2/Fig2_E_npy1r_expression.csv')

npy1r_exp = np.array(df_npy1r['NPY1R'])
rsq = np.array(df_npy1r['rsq'])

# plot NPY1R expression on the brain
toplot = npy1r_exp
brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                  vmin=np.percentile(toplot, 0),
                                  vmax=np.percentile(toplot, 100),
                                  colormap='viridis', customcmap=megcmap,
                                  colorbartitle='NPY1R expression',
                                  surf='inflated')

# correlate NPY1R expression with regional R2 and plot
x = npy1r_exp
y = rsq

corr = stats.spearmanr(x, y)
pvalspin = fcn_megfmri.get_spinp(x, y, corrval=corr[0], nspin=10000,
                                 corrtype='spearman',
                                 lhannot=lhlabels, rhannot=rhlabels)

title = 'spearman r = %1.3f - p (spin) = %1.4f' % (corr[0], pvalspin)
xlab = 'NPY1R expression'
ylab = 'R-squared'
plt.figure()
fcn_megfmri.scatterregplot(x, y, title, xlab, ylab, 60)
