import mayavi
import abagen
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


# path to ready-to-plot data
outpath = '../../data/figures_data/'

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
# save connectivity data as excel and csv files
####################################
# save band-limited MEG connectivity as different sheets of the same excel file
excelFilePath = outpath + 'Fig1/Fig1_A_B_groupFCmeg_aec_orth_schaefer400.xlsx'

# creat list of names for regions
regionList = [f'region {i}' for i in np.arange(400)+1]

# write meg connectivity to excel
with pd.ExcelWriter(excelFilePath) as writer:
    for n, megFC in enumerate(avgFCmeg):
        df_meg = pd.DataFrame(megFC, index=regionList, columns=regionList)
        df_meg.to_excel(writer, sheet_name=bands[n])

# save list of bands
bands_ordering = pd.DataFrame(bands, columns=['bandOrder'])
bands_ordering.to_csv(outpath + 'Fig1/Fig1_A_bands_ordering.csv', index=False)

# save fMRI connectivity as csv file
df_fmri = pd.DataFrame(avgFCmri, index=regionList, columns=regionList)
df_fmri.to_csv(outpath + 'Fig1/Fig1_A_B_groupFCmri_schaefer400.csv')


####################################
# linear regression
####################################
# regional model
rsq, full_pred_fc, corrVal = fcn_megfmri.regional_lreg(megdata=avgFCmeg,
                                                       fmridata=avgFCmri,
                                                       distance=distance,
                                                       correct_dist=False,
                                                       adjusted_rsq=True)

# save predicted fc with regional model
df_pred_fc = pd.DataFrame(full_pred_fc, index=regionList, columns=regionList)
df_pred_fc.to_csv(outpath + '/Fig1/Fig1_C_predicted_fc_regional.csv')

# save R2 from regional model
df_rsq = pd.DataFrame(rsq, columns=['rsq'])
df_rsq.to_csv(outpath + '/Fig1/Fig1_E_rsq_regional.csv', index=False)
df_rsq.to_csv(outpath + '/Fig2/Fig2_A_rsq_regional.csv', index=False)

# save upper triangles of predicted FC from regional model and empirical FC
nnode = full_pred_fc.shape[0]
masku = np.mask_indices(nnode, np.triu, 1)

df_pred_fc_uppertri = pd.DataFrame(full_pred_fc[masku],
                                   columns=['predictedFC_regional'])
df_pred_fc_uppertri['empiricalFC'] = avgFCmri[masku]
df_pred_fc_uppertri.to_csv(outpath +
                           'Fig1/Fig1_C_predicted_fc_regional_uppertri.csv',
                           index=False)

# global model
rsq_g, full_pred_fc_g, corrVal_g = fcn_megfmri.global_lreg(megdata=avgFCmeg,
                                                           fmridata=avgFCmri,
                                                           distance=distance,
                                                           correct_dist=False,
                                                           adjusted_rsq=True)

# save predicted fc with regional model
df_pred_fc = pd.DataFrame(full_pred_fc_g, index=regionList, columns=regionList)
df_pred_fc.to_csv(outpath + 'Fig1/Fig1_D_predicted_fc_global.csv')

# save R2 from regional model (this is only one value)
with open(outpath + 'Fig1/Fig1_E_rsq_global.txt', 'w') as f:
    f.write('%f' % rsq_g)

# save upper triangles of predicted FC from regional model and empirical FC
nnode = full_pred_fc_g.shape[0]
masku = np.mask_indices(nnode, np.triu, 1)

df_pred_fc_uppertri = pd.DataFrame(full_pred_fc_g[masku],
                                   columns=['predictedFC_global'])
df_pred_fc_uppertri['empiricalFC'] = avgFCmri[masku]
df_pred_fc_uppertri.to_csv(outpath +
                           'Fig1/Fig1_D_predicted_fc_global_uppertri.csv',
                           index=False)

####################################
# functional hierarchy
####################################
# use fmri data from same subjects to get fc gradient
grads, lambdas = fcn_megfmri.get_gradients(avgFCmri, ncomp=10)

ncomp = 0
fcgrad = grads[:, ncomp]

df_fcgrad = pd.DataFrame(fcgrad, columns=['fMRI FC Grad1'])
df_fcgrad['rsq'] = rsq
df_fcgrad.to_csv(outpath + '/Fig2/Fig2_B_fcgrad1_rsq.csv', index=False)

####################################
# intrinsic networks
####################################
network_labels = rsnlabels
network_score = pd.DataFrame(data={'R-squared': rsq,
                             'rsn': network_labels})

network_score.to_csv(outpath + 'Fig2/Fig2_C_intrinsic_network_rsq.csv',
                     index=False)

####################################
# BigBrain profile intensity
####################################
# load BigBrain data and normalize them
profileIntensity = np.load('../../data/profileIntensity_schaefer400.npy')
scaler = MinMaxScaler()
profileIntensity_norm = 1-scaler.fit_transform(profileIntensity.T)

df_profileIntensity_norm = pd.DataFrame(profileIntensity_norm)
df_profileIntensity_norm.to_csv(outpath +
                                'Fig2/Fig2_D_normalized_intensity_profile.csv',
                                index=False)

# correlate normalized intensity profile data and regional R2 (spin test)
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

# obtain FDR corrected p-values
multicomp = multitest.multipletests(pvalspin_rsq_intensity.squeeze(),
                                    alpha=0.05, method='fdr_bh')

profileintensity_corr = pd.DataFrame(corr_rsq_intensity.flatten(),
                                     columns=['corrVal'])
profileintensity_corr['pvalSpin'] = pvalspin_rsq_intensity.flatten()
profileintensity_corr['fdr_pvalSpin'] = multicomp[1]

profileintensity_corr.to_csv(outpath + 'Fig2/Fig2_D_profileintensity_rsq_corr.csv',
                             index=False)

####################################
# NPY1R gene
####################################
# to load the data without running abagen
npy1r_exp = np.load('../../data/npy1r_exp.npy')

df_npy1r = pd.DataFrame(npy1r_exp, columns=['NPY1R'])
df_npy1r['rsq'] = rsq

df_npy1r.to_csv(outpath + 'Fig2/Fig2_E_npy1r_expression.csv', index=False)

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

bandContrib.to_csv(outpath + 'Fig3/Fig3_A_bandContribution_dominanceAnalysis.csv',
                   index=False)
bands_ordering = pd.DataFrame(bands, columns=['bandOrder'])
bands_ordering.to_csv(outpath + 'Fig3/Fig3_A_bands_ordering.csv', index=False)

df_percentDominance_adj = pd.DataFrame(percentDominance_adj, columns=bands)
df_percentDominance_adj.to_csv(outpath + 'Fig3/Fig3_D_percentDominance.csv',
                               index=False)

####################################
# percent dominance in intrinsic networks
####################################
percentDominancrsn = pd.DataFrame(percentDominance_adj, columns=bands)
percentDominancrsn['rsn'] = rsnlabels
percentDominancrsn.to_csv(outpath + 'Fig3/Fig3_B_bandContribution_rsn.csv',
                          index=False)

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

df_reordered_bandrsncontrib = pd.DataFrame(reordered_bandrsncontrib, index=bands,
                                           columns=new_order)

df_reordered_bandrsncontrib.to_csv(outpath + 'Fig3/Fig3_B_bandContribution_rsn.csv')

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

maxContribBand = pd.DataFrame(maxContrib, columns=['band'])
maxContribBand['bandIdx'] = maxContribidx

maxContribBand.to_csv(outpath + 'Fig3/Fig3_C_maximumContributingBand.csv',
                      index=False)

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

distance_dep_cv = pd.DataFrame(train_corr, columns=['trainCorr'])
distance_dep_cv['testCorr'] = test_corr
distance_dep_cv.to_csv(outpath + 'Fig4/Fig4_A_regional_cv.csv', index=False)

####################################
# linear regression: subj-level (leave-one-subject out)
####################################
# subject data are not public
subjFCmri = np.load('../../data/subjFCmri_schaefer400.npy')
subjFCmeg = np.load('../../data/subjFCmeg_aec_orth_schaefer400.npy')

train_corr, test_corr = fcn_megfmri.regional_lreg_subj(subjmegdata=subjFCmeg,
                                                       subjfmridata=subjFCmri,
                                                       distance=distance,
                                                       correct_dist=False,
                                                       verbose=True)


subj_cv = pd.DataFrame(train_corr, columns=['trainCorr'])
subj_cv['testCorr'] = test_corr
subj_cv.to_csv(outpath + 'Fig4/Fig4_B_subj_cv.csv', index=False)

####################################
# linear regression: distance correction
####################################
# load data
megdata = np.load('../../data/groupFCmeg_aec_orth_schaefer400.npy.npz')
avgFCmeg = megdata['megfc']
bands = megdata['bands']

# regional model
rsq_new, _, _ = fcn_megfmri.regional_lreg(megdata=avgFCmeg,
                                          fmridata=avgFCmri,
                                          distance=distance,
                                          correct_dist=True,
                                          adjusted_rsq=True)

df_distcorrect = pd.DataFrame(np.array(rsq), columns=['orig_rsq'])
df_distcorrect['new_rsq'] = rsq_new
df_distcorrect['fcgrad'] = fcgrad
df_distcorrect.to_csv(outpath + 'Fig4/Fig4_C_distcorrect.csv', index=False)

####################################
# linear regression: without spatial leakage correction
####################################
# load data
megdata = np.load('../../data/groupFCmeg_aec_schaefer400.npy.npz')
avgFCmeg = megdata['megfc']
bands = megdata['bands']

# regional model
rsq_new, _, _ = fcn_megfmri.regional_lreg(megdata=avgFCmeg,
                                          fmridata=avgFCmri,
                                          distance=distance,
                                          correct_dist=False,
                                          adjusted_rsq=True)

df_noleakage = pd.DataFrame(np.array(rsq), columns=['orig_rsq'])
df_noleakage['new_rsq'] = rsq_new
df_noleakage['fcgrad'] = fcgrad
df_noleakage.to_csv(outpath + 'Fig4/Fig4_D_noleakagecorrection.csv', index=False)

####################################
# linear regression: sLoreta source reconstruction
####################################
# load data
megdata = np.load('../../data/groupFCmeg_aec_orth_schaefer400_sloreta.npy.npz')
avgFCmeg = megdata['megfc']
bands = megdata['bands']

# regional model
rsq_new, _, _ = fcn_megfmri.regional_lreg(megdata=avgFCmeg,
                                          fmridata=avgFCmri,
                                          distance=distance,
                                          correct_dist=False,
                                          adjusted_rsq=True)

df_sloreta = pd.DataFrame(np.array(rsq), columns=['orig_rsq'])
df_sloreta['new_rsq'] = rsq_new
df_sloreta['fcgrad'] = fcgrad
df_sloreta.to_csv(outpath + 'Fig4/Fig4_E_sloreta.csv', index=False)

####################################
# linear regression: PLV connectivity
####################################
# load data
megdata = np.load('../../data/groupFCmeg_plv_schaefer400.npy.npz')
avgFCmeg = megdata['megfc']
bands = megdata['bands']

# regional model
rsq_new, _, _ = fcn_megfmri.regional_lreg(megdata=avgFCmeg,
                                          fmridata=avgFCmri,
                                          distance=distance,
                                          correct_dist=False,
                                          adjusted_rsq=True)

df_plv = pd.DataFrame(np.array(rsq), columns=['orig_rsq'])
df_plv['new_rsq'] = rsq_new
df_plv['fcgrad'] = fcgrad
df_plv.to_csv(outpath + 'Fig4/Fig4_F_plv.csv', index=False)

####################################
# linear regression: Schaefer-200 parcellation
####################################
# load data and calculate everything for Schaefer-200
megdata = np.load('../../data/groupFCmeg_aec_orth_schaefer200.npy.npz')
avgFCmeg = megdata['megfc']
bands = megdata['bands']

avgFCmri = np.load('../../data/groupFCmri_schaefer200.npy')

# load schaefer info
schaeferpath = '../../data/schaefer/'
lhlabels = (schaeferpath + 'HCP/fslr32k/gifti/' +
            'Schaefer2018_200Parcels_7Networks_order_lh.label.gii')
rhlabels = (schaeferpath + 'HCP/fslr32k/gifti/' +
            'Schaefer2018_200Parcels_7Networks_order_rh.label.gii')

# load coordinates and estimate distance
coor = np.loadtxt(schaeferpath + '/Schaefer_200_centres.txt', dtype=str)
coor = coor[:, 1:].astype(float)
distance = sklearn.metrics.pairwise_distances(coor)

# regional model
rsq_new, _, _ = fcn_megfmri.regional_lreg(megdata=avgFCmeg,
                                          fmridata=avgFCmri,
                                          distance=distance,
                                          correct_dist=False,
                                          adjusted_rsq=True)

# use fmri data from same subjects to get fc gradient
grads, lambdas = fcn_megfmri.get_gradients(avgFCmri, ncomp=10)

ncomp = 0
fcgrad = grads[:, ncomp]

df_schefer200= pd.DataFrame(rsq_new, columns=['new_rsq'])
df_schefer200['fcgrad'] = fcgrad
df_schefer200.to_csv(outpath + 'Fig4/Fig4_G_schaefer200.csv', index=False)
