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
# plot Fig4: panel A
####################################
# load regional cross-validation results (distance-dependant cv)
distance_dep_cv = pd.read_csv(figData_path + 'Fig4/Fig4_A_regional_cv.csv')

train_corr = np.array(distance_dep_cv['trainCorr'])
test_corr = np.array(distance_dep_cv['testCorr'])

# plot train and test results on brain surface
toplot = train_corr
brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                  vmin=np.percentile(toplot, 2.5),
                                  vmax=np.percentile(toplot, 97.5),
                                  colormap='viridis', customcmap=megcmap,
                                  colorbartitle='training set - Pearson r',
                                  surf='inflated')

toplot = test_corr
brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                  vmin=np.percentile(toplot, 2.5),
                                  vmax=np.percentile(toplot, 97.5),
                                  colormap='viridis', customcmap=megcmap,
                                  colorbartitle='test set - Pearson r',
                                  surf='inflated')

# correlate train and test results and plot
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
# plot Fig4: panel B
####################################
# load subject-level cross-validation results
subj_cv = pd.read_csv(figData_path + 'Fig4/Fig4_B_subj_cv.csv')

train_corr = np.array(subj_cv['trainCorr'])
test_corr = np.array(subj_cv['testCorr'])

# plot train and test results on brain surface
toplot = train_corr
brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                  vmin=np.percentile(toplot, 2.5),
                                  vmax=np.percentile(toplot, 97.5),
                                  colormap='viridis', customcmap=megcmap,
                                  colorbartitle='training set - Pearson r',
                                  surf='inflated')

toplot = test_corr
brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                  vmin=np.percentile(toplot, 2.5),
                                  vmax=np.percentile(toplot, 97.5),
                                  colormap='viridis', customcmap=megcmap,
                                  colorbartitle='test set - Pearson r',
                                  surf='inflated')

# correlate train and test results and plot
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
# plot Fig4: panel C
####################################
# load distance-corrected results
df_distcorrect = pd.read_csv(figData_path + 'Fig4/Fig4_C_distcorrect.csv')

orig_rsq = np.array(df_distcorrect['orig_rsq'])
new_rsq = np.array(df_distcorrect['new_rsq'])
fcgrad = np.array(df_distcorrect['fcgrad'])

# plot new regional R2 on brain surface
toplot = new_rsq
brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                  vmin=np.percentile(toplot, 2.5),
                                  vmax=np.percentile(toplot, 97.5),
                                  colormap='viridis', customcmap=megcmap,
                                  colorbartitle='R2 - distance corrected',
                                  surf='inflated')

# correlate new R2 with fMRI connectivity gradient and plot
x = fcgrad
y = new_rsq

corr = stats.spearmanr(x, y)
pvalspin = fcn_megfmri.get_spinp(x, y, corrval=corr[0], nspin=10000,
                                 corrtype='spearman',
                                 lhannot=lhlabels, rhannot=rhlabels)

title = 'Spearman r = %1.3f - p (spin) = %1.4f' % (corr[0], pvalspin)
xlab = 'functional hierarchy'
ylab = 'R2 - distance corrected'
plt.figure()
fcn_megfmri.scatterregplot(x, y, title, xlab, ylab, 60)

# correlate new R2 with orig R2 and plot
x = orig_rsq
y = new_rsq

corr = stats.spearmanr(x, y)
pvalspin = fcn_megfmri.get_spinp(x, y, corrval=corr[0], nspin=10000,
                                 corrtype='spearman',
                                 lhannot=lhlabels, rhannot=rhlabels)

title = 'Spearman r = %1.3f - p (spin) = %1.4f' % (corr[0], pvalspin)
xlab = 'R2 - original'
ylab = 'R2 - distance corrected'
plt.figure()
fcn_megfmri.scatterregplot(x, y, title, xlab, ylab, 60)

####################################
# plot Fig4: panel D
####################################
# load no spatial leakage corrected results
df_noleakage = pd.read_csv(figData_path + 'Fig4/Fig4_D_noleakagecorrection.csv')

orig_rsq = np.array(df_noleakage['orig_rsq'])
new_rsq = np.array(df_noleakage['new_rsq'])
fcgrad = np.array(df_noleakage['fcgrad'])

# plot new regional R2 on brain surface
toplot = new_rsq
brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                  vmin=np.percentile(toplot, 2.5),
                                  vmax=np.percentile(toplot, 97.5),
                                  colormap='viridis', customcmap=megcmap,
                                  colorbartitle='R2 - no leakage correction',
                                  surf='inflated')

# correlate new R2 with fMRI connectivity gradient and plot
x = fcgrad
y = new_rsq

corr = stats.spearmanr(x, y)
pvalspin = fcn_megfmri.get_spinp(x, y, corrval=corr[0], nspin=10000,
                                 corrtype='spearman',
                                 lhannot=lhlabels, rhannot=rhlabels)

title = 'Spearman r = %1.3f - p (spin) = %1.4f' % (corr[0], pvalspin)
xlab = 'functional hierarchy'
ylab = 'R2 - no leakage correction'
plt.figure()
fcn_megfmri.scatterregplot(x, y, title, xlab, ylab, 60)

# correlate new R2 with orig R2 and plot
x = orig_rsq
y = new_rsq

corr = stats.spearmanr(x, y)
pvalspin = fcn_megfmri.get_spinp(x, y, corrval=corr[0], nspin=10000,
                                 corrtype='spearman',
                                 lhannot=lhlabels, rhannot=rhlabels)

title = 'Spearman r = %1.3f - p (spin) = %1.4f' % (corr[0], pvalspin)
xlab = 'R2 - original'
ylab = 'R2 - no leakage correction'
plt.figure()
fcn_megfmri.scatterregplot(x, y, title, xlab, ylab, 60)

####################################
# plot Fig4: panel E
####################################
# load sLoreta source reconstruction results
df_sloreta = pd.read_csv(figData_path + 'Fig4/Fig4_E_sloreta.csv')

orig_rsq = np.array(df_sloreta['orig_rsq'])
new_rsq = np.array(df_sloreta['new_rsq'])
fcgrad = np.array(df_sloreta['fcgrad'])

# plot new regional R2 on brain surface
toplot = new_rsq
brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                  vmin=np.percentile(toplot, 2.5),
                                  vmax=np.percentile(toplot, 97.5),
                                  colormap='viridis', customcmap=megcmap,
                                  colorbartitle='R2 - sLoreta',
                                  surf='inflated')

# correlate new R2 with fMRI connectivity gradient and plot
x = fcgrad
y = new_rsq

corr = stats.spearmanr(x, y)
pvalspin = fcn_megfmri.get_spinp(x, y, corrval=corr[0], nspin=10000,
                                 corrtype='spearman',
                                 lhannot=lhlabels, rhannot=rhlabels)

title = 'Spearman r = %1.3f - p (spin) = %1.4f' % (corr[0], pvalspin)
xlab = 'functional hierarchy'
ylab = 'R2 - sLoreta'
plt.figure()
fcn_megfmri.scatterregplot(x, y, title, xlab, ylab, 60)

# correlate new R2 with orig R2 and plot
x = orig_rsq
y = new_rsq

corr = stats.spearmanr(x, y)
pvalspin = fcn_megfmri.get_spinp(x, y, corrval=corr[0], nspin=10000,
                                 corrtype='spearman',
                                 lhannot=lhlabels, rhannot=rhlabels)

title = 'Spearman r = %1.3f - p (spin) = %1.4f' % (corr[0], pvalspin)
xlab = 'R2 - original'
ylab = 'R2 - sLoreta'
plt.figure()
fcn_megfmri.scatterregplot(x, y, title, xlab, ylab, 60)

####################################
# plot Fig4: panel F
####################################
# load PLV connectivity results
df_plv = pd.read_csv(figData_path + 'Fig4/Fig4_F_plv.csv')

orig_rsq = np.array(df_plv['orig_rsq'])
new_rsq = np.array(df_plv['new_rsq'])
fcgrad = np.array(df_plv['fcgrad'])

# plot new regional R2 on brain surface
toplot = new_rsq
brains = fcn_megfmri.plot_conte69(toplot, lhlabels, rhlabels,
                                  vmin=np.percentile(toplot, 2.5),
                                  vmax=np.percentile(toplot, 97.5),
                                  colormap='viridis', customcmap=megcmap,
                                  colorbartitle='R2 - PLV',
                                  surf='inflated')

# correlate new R2 with fMRI connectivity gradient and plot
x = fcgrad
y = new_rsq

corr = stats.spearmanr(x, y)
pvalspin = fcn_megfmri.get_spinp(x, y, corrval=corr[0], nspin=10000,
                                 corrtype='spearman',
                                 lhannot=lhlabels, rhannot=rhlabels)

title = 'Spearman r = %1.3f - p (spin) = %1.4f' % (corr[0], pvalspin)
xlab = 'functional hierarchy'
ylab = 'R2 - PLV'
plt.figure()
fcn_megfmri.scatterregplot(x, y, title, xlab, ylab, 60)

# correlate new R2 with orig R2 and plot
x = orig_rsq
y = new_rsq

corr = stats.spearmanr(x, y)
pvalspin = fcn_megfmri.get_spinp(x, y, corrval=corr[0], nspin=10000,
                                 corrtype='spearman',
                                 lhannot=lhlabels, rhannot=rhlabels)

title = 'Spearman r = %1.3f - p (spin) = %1.4f' % (corr[0], pvalspin)
xlab = 'R2 - original'
ylab = 'R2 - PLV'
plt.figure()
fcn_megfmri.scatterregplot(x, y, title, xlab, ylab, 60)

####################################
# plot Fig4: panel G
####################################
# load Schaefer-200 results
df_schaefer200 = pd.read_csv(figData_path + 'Fig4/Fig4_G_schaefer200.csv')

new_rsq = np.array(df_schaefer200['new_rsq'])
fcgrad = np.array(df_schaefer200['fcgrad'])

# load plotting info for Schaefer-200
schaeferpath = '../../data/schaefer/'
lhlabels_200 = (schaeferpath + 'HCP/fslr32k/gifti/' +
                'Schaefer2018_200Parcels_7Networks_order_lh.label.gii')
rhlabels_200 = (schaeferpath + 'HCP/fslr32k/gifti/' +
                'Schaefer2018_200Parcels_7Networks_order_rh.label.gii')

# plot new regional R2 on brain surface
toplot = new_rsq
brains = fcn_megfmri.plot_conte69(toplot, lhlabels_200, rhlabels_200,
                                  vmin=np.percentile(toplot, 2.5),
                                  vmax=np.percentile(toplot, 97.5),
                                  colormap='viridis', customcmap=megcmap,
                                  colorbartitle='R2 - Schaefer-200',
                                  surf='inflated')

# correlate new R2 with fMRI connectivity gradient and plot
x = fcgrad
y = new_rsq

corr = stats.spearmanr(x, y)
pvalspin = fcn_megfmri.get_spinp(x, y, corrval=corr[0], nspin=10000,
                                 corrtype='spearman',
                                 lhannot=lhlabels_200, rhannot=rhlabels_200)

title = 'Spearman r = %1.3f - p (spin) = %1.4f' % (corr[0], pvalspin)
xlab = 'functional hierarchy'
ylab = 'R2 - Schaefer-200'
plt.figure()
fcn_megfmri.scatterregplot(x, y, title, xlab, ylab, 60)
