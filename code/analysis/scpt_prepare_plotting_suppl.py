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

rsq_band.to_csv(outpath + 'FigS1/FigS1_band_specific_rsq.csv', index=False)

####################################
# structure-function coupling
####################################
sc_consensus = np.load('../../data/consensusSC_wei_Schaefer400.npy')

scfc = []
for node in range(avgFCmri.shape[0]):
    X = avgFCmri[:, node]
    X = np.delete(X, node)

    Y = sc_consensus[:, node]
    Y = np.delete(Y, node)

    nnzidx = np.where(Y != 0)

    scfc.append(stats.spearmanr(X[nnzidx], Y[nnzidx])[0])

# regional model
rsq, _, _ = fcn_megfmri.regional_lreg(megdata=avgFCmeg,
                                      fmridata=avgFCmri,
                                      distance=distance,
                                      correct_dist=False,
                                      adjusted_rsq=True)

df_scfc = pd.DataFrame(scfc, columns=['scfc_coupling'])
df_scfc['rsq'] = rsq
df_scfc.to_csv(outpath + 'FigS2/FigS2_scfc_coupling.csv', index=False)

####################################
# ctf metric
####################################
# prepare data for ctf lcmv
ctf_lcmv = np.load('../../data/avg_peak_err_ctf_resolutionMatrix_lcmv_Schaefer400.npy')

# regional model
rsq, _, _ = fcn_megfmri.regional_lreg(megdata=avgFCmeg,
                                      fmridata=avgFCmri,
                                      distance=distance,
                                      correct_dist=False,
                                      adjusted_rsq=True)

df_ctf_lcmv = pd.DataFrame(ctf_lcmv, columns=['ctf_lcmv'])
df_ctf_lcmv['rsq_lcmv (orig)'] = rsq
df_ctf_lcmv.to_csv(outpath + 'FigS3/FigS3_A_ctf_lcmv.csv', index=False)

# prepare data for ctf sloreta
ctf_sloreta = np.load('../../data/avg_peak_err_ctf_resolutionMatrix_sloreta_Schaefer400.npy')

# load MEG data with sLoreta source reconstruction to calculate sLoreta R2
megdata_sloreta = np.load('../../data/groupFCmeg_aec_orth_schaefer400_sloreta.npy.npz')
avgFCmeg_slortea = megdata_sloreta['megfc']

# regional model
rsq_sloreta, _, _ = fcn_megfmri.regional_lreg(megdata=avgFCmeg_slortea,
                                              fmridata=avgFCmri,
                                              distance=distance,
                                              correct_dist=False,
                                              adjusted_rsq=True)

df_ctf_sloreta = pd.DataFrame(ctf_sloreta, columns=['ctf_sloreta'])
df_ctf_sloreta['rsq_sloreta'] = rsq_sloreta
df_ctf_sloreta.to_csv(outpath + 'FigS3/FigS3_B_ctf_sloreta.csv', index=False)

####################################
# SNR
####################################
avg_snr = np.load('../../data/avgsnr_schaefer400.npy')

df_snr = pd.DataFrame(avg_snr, columns=['SNR'])
df_snr['rsq'] = rsq
df_snr.to_csv(outpath + 'FigS4/FigS4_snr.csv', index=False)

####################################
# pairwise similarity of MEG connectivity
####################################
mask = np.mask_indices(400, np.triu, 1)
meg_uppertriangle = [bandmegfc[mask] for bandmegfc in avgFCmeg]

df_meg_uppertriangle = pd.DataFrame(np.array(meg_uppertriangle).T,
                                    columns=bands)
df_meg_uppertriangle.to_csv(outpath +
                            'FigS5/FigS5_meg_connectivity_uppertriangle.csv',
                            index=False)
                            
ax = sns.heatmap(np.corrcoef(np.array(meg_uppertriangle)), cmap=cmap_seq,
                 vmin=-0.1,
                 vmax=np.max(np.corrcoef(np.array(meg_uppertriangle))),
                 linewidth=0.5,
                 xticklabels=bands, yticklabels=bands, square=True)
ax.axes.set_title('band MEG fc correlation')
ax.figure.set_figwidth(7)
ax.figure.set_figheight(6)
plt.tight_layout()
