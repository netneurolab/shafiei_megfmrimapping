import copy
import scipy
import sklearn
import numpy as np
import nibabel as nib
import seaborn as sns
import sklearn.metrics
import palettable as pal
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
from mapalign.embed import compute_diffusion_map
from netneurotools import stats as netneurostats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap


def fit_exponential(x, a, b, c):
    return a * np.exp(-b * x) + c


def remove_distance(fcdata, distance):
    nnode = distance.shape[0]
    mask = np.mask_indices(nnode, np.triu, 1)

    fcdata_masked = fcdata[mask]
    distance_masked = distance[mask]

    x = distance_masked
    y = fcdata_masked

    popt, pcov = curve_fit(fit_exponential, x, y, bounds=(-1, 2))
    modeled_y = fit_exponential(x, *popt)

    temp = y - modeled_y
    tempConn = np.zeros((nnode, nnode))
    tempConn[mask] = temp
    tempConn = tempConn + tempConn.T
    np.fill_diagonal(tempConn, 1)
    residConnectivity = tempConn
    residConnectivityDistCorr = stats.pearsonr(x, temp)[0]

    return residConnectivity, residConnectivityDistCorr


def regional_lreg(megdata, fmridata, distance, correct_dist=True,
                  adjusted_rsq=True):
    rsq = []
    rsq_adj = []
    corrVal = []
    full_pred_fc = np.empty_like(fmridata)
    nnode = distance.shape[0]

    if correct_dist:
        mri_distcorrect = remove_distance(fmridata, distance)[0]
        meg_distcorrect = [remove_distance(megdata[band], distance)[0]
                           for band in range(len(megdata))]

    for node in range(nnode):
        if correct_dist:
            avgFCmeg_node = [meg_distcorrect[n][:, node]
                             for n in range(len(megdata))]
            avgFCmeg_node = np.array(avgFCmeg_node).T

            X = np.delete(avgFCmeg_node, node, axis=0)

            Y = mri_distcorrect[:, node]
            Y = np.delete(Y, node)
        else:
            avgFCmeg_node = [megdata[band][:, node]
                             for band in range(len(megdata))]
            avgFCmeg_node = np.array(avgFCmeg_node).T

            X = np.delete(avgFCmeg_node, node, axis=0)

            Y = fmridata[:, node]
            Y = np.delete(Y, node)

        model = LinearRegression(fit_intercept=True, normalize=True).fit(X, Y)
        pred_Y = model.predict(X)

        resid_sumsquare = ((Y - pred_Y) ** 2).sum()
        total_sumsquare = ((Y - np.mean(Y)) ** 2).sum()
        rsquared = 1 - (float(resid_sumsquare)) / total_sumsquare

        adjusted_rsquared = 1 - (1 - rsquared) * \
            (len(Y) - 1) / (len(Y) - X.shape[1] - 1)

        rsq.append(rsquared)
        rsq_adj.append(adjusted_rsquared)

        pred_Y_full = list(pred_Y)
        pred_Y_full.insert(node, 1)
        full_pred_fc[node, :] = pred_Y_full

        corrVal.append(stats.pearsonr(Y.flatten(), pred_Y.flatten())[0])

    if adjusted_rsq:
        return rsq_adj, full_pred_fc, corrVal
    else:
        return rsq, full_pred_fc, corrVal


def global_lreg(megdata, fmridata, distance, correct_dist=True,
                adjusted_rsq=True):
    rsq = []
    rsq_adj = []
    corrVal = []
    full_pred_fc = np.empty_like(fmridata)

    nnode = distance.shape[0]
    mask = np.mask_indices(nnode, np.triu, 1)

    if correct_dist:
        mri_distcorrect = remove_distance(fmridata, distance)[0]
        meg_distcorrect = [remove_distance(megdata[band], distance)[0]
                           for band in range(len(megdata))]

        masked_fmri = mri_distcorrect[mask]

        masked_meg = [meg_distcorrect[n][mask] for n in range(len(megdata))]
        masked_meg = np.array(masked_meg).T

        X = masked_meg
        Y = masked_fmri
    else:
        masked_fmri = fmridata[mask]

        masked_meg = [megdata[n][mask] for n in range(len(megdata))]
        masked_meg = np.array(masked_meg).T

        X = masked_meg
        Y = masked_fmri

    model = LinearRegression(fit_intercept=True, normalize=True).fit(X, Y)
    pred_Y = model.predict(X)

    resid_sumsquare = ((Y - pred_Y) ** 2).sum()
    total_sumsquare = ((Y - np.mean(Y)) ** 2).sum()
    rsquared = 1 - (float(resid_sumsquare)) / total_sumsquare

    adjusted_rsquared = 1 - (1 - rsquared) * \
        (len(Y) - 1) / (len(Y) - X.shape[1] - 1)

    corrVal = stats.pearsonr(Y.flatten(), pred_Y.flatten())[0]

    temp = np.zeros((nnode, nnode))
    temp[mask] = pred_Y
    full_pred_fc = temp + temp.T
    np.fill_diagonal(full_pred_fc, 1)

    if adjusted_rsq:
        return adjusted_rsquared, full_pred_fc, corrVal
    else:
        return rsquared, full_pred_fc, corrVal


def get_percent_dominance(megdata, fmridata, distance, correct_dist=True,
                          adjusted_rsq=True):
    percentDominance = []
    nnode = np.shape(distance)[0]

    if correct_dist:
        mri_distcorrect = remove_distance(fmridata, distance)[0]
        meg_distcorrect = [remove_distance(megdata[band], distance)[0]
                           for band in range(len(megdata))]

    for node in range(nnode):
        if correct_dist:
            avgFCmeg_node = [meg_distcorrect[n][:, node]
                             for n in range(len(megdata))]
            avgFCmeg_node = np.array(avgFCmeg_node).T

            X = np.delete(avgFCmeg_node, node, axis=0)

            Y = mri_distcorrect[:, node]
            Y = np.delete(Y, node)
        else:
            avgFCmeg_node = [megdata[band][:, node]
                             for band in range(len(megdata))]
            avgFCmeg_node = np.array(avgFCmeg_node).T

            X = np.delete(avgFCmeg_node, node, axis=0)

            Y = fmridata[:, node]
            Y = np.delete(Y, node)


        if adjusted_rsq:
            model_metrics, model_rsq = netneurostats.get_dominance_stats(X, Y,
                                        use_adjusted_r_sq=True)
        else:
            model_metrics, model_rsq = netneurostats.get_dominance_stats(X, Y,
                                        use_adjusted_r_sq=False)
        dom_ratio = model_metrics['total_dominance']/model_metrics['full_r_sq']
        percentDominance.append(dom_ratio*100)

    return np.array(percentDominance)


def get_gradients(connectmat, ncomp):
    threshMat = connectmat.copy()
    np.fill_diagonal(threshMat, 0)

    # Generate percentile thresholds for 90th percentile
    perc = np.array([np.percentile(x, 90) for x in threshMat])

    # Threshold each row of the matrix by setting values below 90th perc to 0
    for i in range(threshMat.shape[0]):
        threshMat[i, threshMat[i, :] < perc[i]] = 0

    # Count negative values per row
    neg_values = np.array([sum(threshMat[i, :] < 0)
                          for i in range(threshMat.shape[0])])
    print('Negative values occur in %d rows' % sum(neg_values > 0))

    # remove negative ones
    threshMat[threshMat < 0] = 0

    cosSimilarity = sklearn.metrics.pairwise.cosine_similarity(threshMat)
    dme = compute_diffusion_map(cosSimilarity, n_components=ncomp,
                                return_result=True)

    # lambdas
    lambdas = dme[1]['lambdas']

    # gradients
    grads = dme[0]

    return grads, lambdas


def get_gifti_centroids(surfaces, lhannot, rhannot):
    lhsurface, rhsurface = [nib.load(s) for s in surfaces]

    centroids, hemiid = [], []
    for n, (annot, surf) in enumerate(zip([lhannot, rhannot],
                                          [lhsurface, rhsurface])):
        vert, face = [d.data for d in surf.darrays]
        labels = np.squeeze(nib.load(annot).darrays[0].data)

        for lab in np.unique(labels):
            if lab == 0:
                continue
            coords = np.atleast_2d(vert[labels == lab].mean(axis=0))
            roi = vert[np.argmin(cdist(vert, coords), axis=0)[0]]
            centroids.append(roi)
            hemiid.append(n)

    centroids = np.row_stack(centroids)
    hemiid = np.asarray(hemiid)

    return centroids, hemiid


def get_spinp(x, y, corrval, nspin, lhannot, rhannot, corrtype):
    surf_path = ('../../data/surfaces/')
    surfaces = [surf_path + 'L.sphere.32k_fs_LR.surf.gii',
                surf_path + 'R.sphere.32k_fs_LR.surf.gii']
    lhannot = lhannot
    rhannot = rhannot

    centroids, hemiid = get_gifti_centroids(surfaces, lhannot,
                                                         rhannot)
    spins = netneurostats.gen_spinsamples(centroids, hemiid,
                                          n_rotate=nspin, seed=272)

    permuted_r = np.zeros((nspin, 1))
    for spin in range(nspin):
        if corrtype == 'spearman':
            permuted_r[spin] = scipy.stats.spearmanr(x[spins[:, spin]], y)[0]
        elif corrtype == 'pearson':
            permuted_r[spin] = scipy.stats.pearsonr(x[spins[:, spin]], y)[0]

    permmean = np.mean(permuted_r)
    pvalspin = (len(np.where(abs(permuted_r - permmean) >=
                             abs(corrval - permmean))[0])+1)/(nspin+1)
    return pvalspin


def get_spinidx(nspin, lhannot, rhannot):
    surf_path = ('../../data/surfaces/')
    surfaces = [surf_path + 'L.sphere.32k_fs_LR.surf.gii',
                surf_path + 'R.sphere.32k_fs_LR.surf.gii']
    lhannot = lhannot
    rhannot = rhannot

    centroids, hemiid = get_gifti_centroids(surfaces, lhannot,
                                                         rhannot)
    spins = netneurostats.gen_spinsamples(centroids, hemiid,
                                          n_rotate=nspin, seed=272)
    return spins


def train_test_split_distance(X, Y, coords, train_pct=.75, sourceNode='random'):
    distance = sklearn.metrics.pairwise_distances(coords)
    if sourceNode == 'random':
        sourceNode = np.random.choice(range(0, len(coords), 1))
    else:
        # print(sourceNode)
        sourceNode = sourceNode
    distances = distance[sourceNode, :]
    idx = np.argsort(distances)
    train_idx = idx[:int(np.floor(train_pct * len(coords)))]
    test_idx = idx[int(np.floor(train_pct * len(coords))):]
    X_train = X[train_idx, :]
    Y_train = Y[train_idx]

    X_test = X[test_idx, :]
    Y_test = Y[test_idx]

    return X_train, X_test, Y_train, Y_test


def regional_lreg_cv(megdata, fmridata, distance, coor, correct_dist=True,
                     train_pct=0.75, verbose=False):

    nnode = distance.shape[0]
    nsplit = nnode-1

    corr_train_avg = []
    corr_test_avg = []

    if correct_dist:
        mri_distcorrect = remove_distance(fmridata, distance)[0]
        meg_distcorrect = [remove_distance(megdata[band], distance)[0]
                           for band in range(len(megdata))]

    for node in range(nnode):
        if correct_dist:
            avgFCmeg_node = [meg_distcorrect[n][:, node]
                             for n in range(len(megdata))]
            avgFCmeg_node = np.array(avgFCmeg_node).T

            X = np.delete(avgFCmeg_node, node, axis=0)

            Y = mri_distcorrect[:, node]
            Y = np.delete(Y, node)
        else:
            avgFCmeg_node = [megdata[band][:, node]
                             for band in range(len(megdata))]
            avgFCmeg_node = np.array(avgFCmeg_node).T

            X = np.delete(avgFCmeg_node, node, axis=0)

            Y = fmridata[:, node]
            Y = np.delete(Y, node)

        # cross-validation
        corr_train = []
        corr_test = []

        mod_coor = coor.copy()
        mod_coor = np.delete(mod_coor, node, axis=0)

        for split in range(nsplit):
            X_train, X_test, Y_train, Y_test = train_test_split_distance(
                X, Y, mod_coor, train_pct=train_pct, sourceNode='random')

            sc_x = StandardScaler()
            sc_x.fit(X_train)
            X_train = sc_x.transform(X_train)
            X_test = sc_x.transform(X_test)
            sc_y = StandardScaler()
            sc_y.fit(Y_train.reshape(-1, 1))
            Y_train = sc_y.transform(Y_train.reshape(-1, 1))
            Y_test = sc_y.transform(Y_test.reshape(-1, 1))

            model = LinearRegression(fit_intercept=True, normalize=False).fit(
                X_train, Y_train)

            pred_Y_train = model.predict(X_train)
            pred_Y_test = model.predict(X_test)
            corr_train.append(stats.pearsonr(Y_train.flatten(),
                                            pred_Y_train.flatten())[0])
            corr_test.append(stats.pearsonr(Y_test.flatten(),
                                            pred_Y_test.flatten())[0])

        corr_train_avg.append(np.mean(corr_train))
        corr_test_avg.append(np.mean(corr_test))

        if verbose:
            print('\nCV node %s of %s nodes done!' % (node+1, nnode))

    return np.array(corr_train_avg), np.array(corr_test_avg)


def regional_lreg_subj(subjmegdata, subjfmridata, distance, correct_dist=True,
                       verbose=False):
    # iterate over subjects and predict R2 for each
    nnode = distance.shape[0]
    nsubj = subjfmridata.shape[2]
    nband = len(subjmegdata)

    subjCorrVal = []
    groupCorrVal = []

    for subj in range(nsubj):
        leaveoutsubjFCmri = subjfmridata[:, :, subj]
        leaveoutsubjFCmeg = []
        for band in range(nband):
            leaveoutsubjFCmeg.append(subjmegdata[band][:, :, subj])

        tempFCmri = subjfmridata.copy()
        tempFCmri[:, :, subj] = np.nan
        avgFCmriSubjects = np.nanmean(tempFCmri, 2)

        tempFCmeg = copy.deepcopy(subjmegdata)
        for band in range(nband):
            tempFCmeg[band][:, :, subj] = np.nan
        avgFCmegSubjects = [np.nanmean(tempFCmeg[band], axis=2)
                            for band in range(nband)]

        if correct_dist:
            mask = np.mask_indices(nnode, np.triu, 1)

            maskedFCmeg = [avgFCmegSubjects[band][mask]
                           for band in range(nband)]
            allconnectivityGroup = maskedFCmeg.copy()
            allconnectivityGroup.append(avgFCmriSubjects[mask])

            maskedFCmegSubj = [leaveoutsubjFCmeg[band][mask]
                               for band in range(nband)]
            allconnectivitySubj = maskedFCmegSubj.copy()
            allconnectivitySubj.append(leaveoutsubjFCmri[mask])

            residConnectivityGroup = []
            residConnectivitySubj = []
            for n in range(len(allconnectivityGroup)):
                x = distance[mask]
                y = allconnectivityGroup[n]

                # fit the exponential model on group data (train data) and use
                # the same fit to correct subj data (test data) for distance
                popt, pcov = curve_fit(fit_exponential, x, y, bounds=(-1, 2))
                modeled_y = fit_exponential(x, *popt)

                temp = y - modeled_y
                tempConn = np.zeros((nnode, nnode))
                tempConn[mask] = temp
                tempConn = tempConn + tempConn.T
                np.fill_diagonal(tempConn, 1)
                residConnectivityGroup.append(tempConn)

                y_subj = allconnectivitySubj[n]
                temp = y_subj - modeled_y
                tempConn = np.zeros((nnode, nnode))
                tempConn[mask] = temp
                tempConn = tempConn + tempConn.T
                np.fill_diagonal(tempConn, 1)
                residConnectivitySubj.append(tempConn)

            subjmri = residConnectivitySubj[-1]
            groupmri = residConnectivityGroup[-1]

            subjmeg = residConnectivitySubj[:nband]
            groupmeg = residConnectivityGroup[:nband]
        else:
            subjmri = leaveoutsubjFCmri
            groupmri = avgFCmriSubjects

            subjmeg = leaveoutsubjFCmeg
            groupmeg = avgFCmegSubjects

        corrVal_nodes_group = []
        corrVal_nodes_subj = []
        for node in range(nnode):
            avgFreqBandFC = [groupmeg[band][:, node] for band in range(nband)]
            avgFreqBandFC = np.array(avgFreqBandFC).T

            X = np.delete(avgFreqBandFC, node, axis=0)

            Y = groupmri[:, node]
            Y = np.delete(Y, node)

            X_subj = [subjmeg[band][:, node] for band in range(nband)]
            X_subj = np.array(X_subj).T
            X_subj = np.delete(X_subj, node, axis=0)

            Y_subj = subjmri[:, node]
            Y_subj = np.delete(Y_subj, node)

            sc_x = StandardScaler()
            sc_x.fit(X)
            X = sc_x.transform(X)
            X_subj = sc_x.transform(X_subj)
            sc_y = StandardScaler()
            sc_y.fit(Y.reshape(-1, 1))
            Y = sc_y.transform(Y.reshape(-1, 1))
            Y_subj = sc_y.transform(Y_subj.reshape(-1, 1))

            model = LinearRegression(fit_intercept=True,
                                     normalize=False).fit(X, Y)

            Y_pred_subj = model.predict(X_subj)
            corrVal = stats.pearsonr(Y_pred_subj.flatten(),
                                     Y_subj.flatten())[0]
            corrVal_nodes_subj.append(corrVal)

            Y_pred_group = model.predict(X)
            corrVal = stats.pearsonr(Y_pred_group.flatten(), Y.flatten())[0]
            corrVal_nodes_group.append(corrVal)

        subjCorrVal.append(corrVal_nodes_subj)
        groupCorrVal.append(corrVal_nodes_group)

        if verbose:
            print('\nSubj %s of %s subjects done!' % (subj+1, nsubj))

    subjCorrVal = np.array(subjCorrVal)
    groupCorrVal = np.array(groupCorrVal)

    return np.mean(groupCorrVal, axis=0), np.mean(subjCorrVal, axis=0)


def scatterregplot(x, y, title, xlab, ylab, pointsize):
    myplot = sns.scatterplot(x, y,
                             facecolor=np.array([128/255, 128/255, 128/255]),
                             legend=False, rasterized=True)
    sns.regplot(x, y, scatter=False, ax=myplot,
                line_kws=dict(color='k'))
    sns.despine(ax=myplot, trim=False)
    myplot.axes.set_title(title)
    myplot.axes.set_xlabel(xlab)
    myplot.axes.set_ylabel(ylab)
    myplot.figure.set_figwidth(5)
    myplot.figure.set_figheight(5)
    return myplot


def plot_conte69(data, lhlabel, rhlabel, surf='midthickness',
                 vmin=None, vmax=None, colormap='viridis', customcmap=None,
                 colorbar=True, num_labels=4, orientation='horizontal',
                 colorbartitle=None, backgroundcolor=(1, 1, 1),
                 foregroundcolor=(0, 0, 0), **kwargs):

    """
    Plots surface `data` on Conte69 Atlas

    (This is a modified version of plotting.plot_conte69 from netneurotools.
     This version will be merged with the one on netneurotools in future.)

    Parameters
    ----------
    data : (N,) array_like
        Surface data for N parcels
    lhlabel : str
        Path to .gii file (generic GIFTI file) containing labels to N/2 parcels
        on the left hemisphere
    rhlabel : str
        Path to .gii file (generic GIFTI file) containing labels to N/2 parcels
        on the right hemisphere
    surf : {'midthickness', 'inflated', 'vinflated'}, optional
        Type of brain surface. Default: 'midthickness'
    vmin : float, optional
        Minimum value to scale the colormap. If None, the min of the data will
        be used. Default: None
    vmax : float, optional
        Maximum value to scale the colormap. If None, the max of the data will
        be used. Default: None
    colormap : str, optional
        Any colormap from matplotlib. Default: 'viridis'
    colorbar : bool, optional
        Wheter to display a colorbar. Default: True
    num_labels : int, optional
        The number of labels to display on the colorbar.
        Available only if colorbar=True. Default: 4
    orientation : str, optional
        Defines the orientation of colorbar. Can be 'horizontal' or 'vertical'.
        Available only if colorbar=True. Default: 'horizontal'
    colorbartitle : str, optional
        The title of colorbar. Available only if colorbar=True. Default: None
    backgroundcolor : tuple of float values with RGB code in [0, 1], optional
        Defines the background color. Default: (1, 1, 1)
    foregroundcolor : tuple of float values with RGB code in [0, 1], optional
        Defines the foreground color (e.g., colorbartitle color).
        Default: (0, 0, 0)
    kwargs : key-value mapping
        Keyword arguments for `mayavi.mlab.triangular_mesh()`

    Returns
    -------
    scene : mayavi.Scene
        Scene object containing plot
    """

    from netneurotools.datasets import fetch_conte69
    try:
        from mayavi import mlab
    except ImportError:
        raise ImportError('Cannot use plot_conte69() if mayavi is not '
                          'installed. Please install mayavi and try again.')

    opts = dict()
    opts.update(**kwargs)

    try:
        surface = fetch_conte69()[surf]
    except KeyError:
        raise ValueError('Provided surf "{}" is not valid. Must be one of '
                         '[\'midthickness\', \'inflated\', \'vinflated\']'
                         .format(surf))
    lhsurface, rhsurface = [nib.load(s) for s in surface]

    lhlabels = nib.load(lhlabel).darrays[0].data
    rhlabels = nib.load(rhlabel).darrays[0].data
    lhvert, lhface = [d.data for d in lhsurface.darrays]
    rhvert, rhface = [d.data for d in rhsurface.darrays]

    # add NaNs for subcortex
    data = np.append(np.nan, data)

    # get lh and rh data
    lhdata = np.squeeze(data[lhlabels.astype(int)])
    rhdata = np.squeeze(data[rhlabels.astype(int)])

    # plot
    lhplot = mlab.figure()
    rhplot = mlab.figure()
    lhmesh = mlab.triangular_mesh(lhvert[:, 0], lhvert[:, 1], lhvert[:, 2],
                                  lhface, figure=lhplot, colormap=colormap,
                                  mask=np.isnan(lhdata), scalars=lhdata,
                                  vmin=vmin, vmax=vmax, **opts)
    lhmesh.module_manager.scalar_lut_manager.lut.nan_color = [0.863, 0.863,
                                                              0.863, 1]
    lhmesh.update_pipeline()
    if type(customcmap) != str:
        lut = lhmesh.module_manager.scalar_lut_manager.lut.table.to_array()
        lut[:, :3] = customcmap.colors * 255
        lhmesh.module_manager.scalar_lut_manager.lut.table = lut
        mlab.draw()
    if colorbar is True:
        mlab.colorbar(title=colorbartitle, nb_labels=num_labels,
                      orientation=orientation)
    rhmesh = mlab.triangular_mesh(rhvert[:, 0], rhvert[:, 1], rhvert[:, 2],
                                  rhface, figure=rhplot, colormap=colormap,
                                  mask=np.isnan(rhdata), scalars=rhdata,
                                  vmin=vmin, vmax=vmax, **opts)
    rhmesh.module_manager.scalar_lut_manager.lut.nan_color = [0.863, 0.863,
                                                              0.863, 1]
    rhmesh.update_pipeline()
    if type(customcmap) != str:
        lut = rhmesh.module_manager.scalar_lut_manager.lut.table.to_array()
        lut[:, :3] = customcmap.colors * 255
        rhmesh.module_manager.scalar_lut_manager.lut.table = lut
        mlab.draw()
    if colorbar is True:
        mlab.colorbar(title=colorbartitle, nb_labels=num_labels,
                      orientation=orientation)
    mlab.view(azimuth=180, elevation=90, distance=450, figure=lhplot)
    mlab.view(azimuth=180, elevation=-90, distance=450, figure=rhplot)

    mlab.figure(bgcolor=backgroundcolor, fgcolor=foregroundcolor,
                figure=lhplot)
    mlab.figure(bgcolor=backgroundcolor, fgcolor=foregroundcolor,
                figure=rhplot)

    return lhplot, rhplot


def make_colormaps():
    cmap_seq = LinearSegmentedColormap.from_list('mycmap', list(reversed(
            np.array(pal.cmocean.sequential.Matter_20.mpl_colors[:-1]))))

    cmap_seq_r = LinearSegmentedColormap.from_list('mycmap', list(
                   np.array(pal.cmocean.sequential.Matter_20.mpl_colors[:-1])))

    cmap_seq_v2 = LinearSegmentedColormap.from_list('mycmap', list(reversed(
                np.array(pal.cartocolors.sequential.SunsetDark_7.mpl_colors))))

    cmap_seq_v2_disc = ListedColormap(list(reversed(
                np.array(pal.cartocolors.sequential.SunsetDark_6.mpl_colors))))

    # test = np.abs(np.random.rand(5, 12))
    # plt.figure()
    # plt.imshow(test, interpolation='nearest', cmap=cmap_seq)
    # plt.colorbar()

    colors = np.vstack([np.array(cmap_seq(i)[:3]) for i in range(256)])
    megcmap = ListedColormap(colors)

    colors = np.vstack([np.array(cmap_seq_v2(i)[:3]) for i in range(256)])
    megcmap2 = ListedColormap(colors)

    colors = np.vstack([np.array(cmap_seq_v2_disc(i)[:3]) for i in range(6)])
    categ_cmap = ListedColormap(np.vstack((np.ones((43, 3)) * colors[0],
                                           np.ones((43, 3)) * colors[1],
                                           np.ones((43, 3)) * colors[2],
                                           np.ones((43, 3)) * colors[3],
                                           np.ones((42, 3)) * colors[4],
                                           np.ones((42, 3)) * colors[5])))

    return cmap_seq, cmap_seq_r, megcmap, megcmap2, categ_cmap


def make_colormap_rsn():
    colors = sns.color_palette('hls', n_colors=7)
    categ_cmap = ListedColormap(np.vstack((np.ones((37, 3)) * np.array(colors[0]),
                                           np.ones((37, 3)) * np.array(colors[1]),
                                           np.ones((37, 3)) * np.array(colors[2]),
                                           np.ones((37, 3)) * np.array(colors[3]),
                                           np.ones((37, 3)) * np.array(colors[4]),
                                           np.ones((37, 3)) * np.array(colors[5]),
                                           np.ones((34, 3)) * np.array(colors[6]))))

    return categ_cmap
