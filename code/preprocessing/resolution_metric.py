import os
import time
import nibabel as nib
import numpy as np
import scipy.io as sio


hcp_dir = '/path/to/megdata/and/results/';
parcellationDir = ('/path/to/SchaeferParcellation/')

# load subj list
subjList = sio.loadmat(os.path.join(hcp_dir, 'myMEGList.mat'))['myMEG']
subjList = [subjList[i][0][0][4:] for i in range(len(subjList))]

# load parcellation data
lhlabel = nib.load(parcellationDir + 'Schaefer400_L.4k.label.gii')
rhlabel = nib.load(parcellationDir + 'Schaefer400_R.4k.label.gii')

parcels = np.concatenate((lhlabel.darrays[0].data, rhlabel.darrays[0].data))

parcelIDs = np.unique(parcels)
parcelIDs = np.delete(parcelIDs, 0)

functions = ['psf', 'ctf']
metric = 'peak_err'
dataTypes = ['resolutionMatrix_sloreta', 'resolutionMatrix_lcmv']
vertexData = 'resolutionMatrix_coordinates'
outpath = os.path.join(hcp_dir, 'brainstormResults/resolutionMatrix_sloreta/')

for dataType in dataTypes[:1]:
    for function in functions:
        group_metric_vertex = []
        group_metric_parcel = []
        for s, subj in enumerate(subjList):
            startTime = time.time()
            path_to_file = os.path.join(hcp_dir, 'brainstormResults', dataType,
                                        subj, (subj + '_resMat.mat'))
            subjFile = sio.loadmat(path_to_file)
            resmat = subjFile['resolutionMat']

            path_to_coord= os.path.join(hcp_dir, 'brainstormResults', vertexData,
                                        subj, (subj + '_vertexCoor.mat'))
            subjCoor = sio.loadmat(path_to_coord)
            locations = subjCoor['vertices']

            ### "locerr" is calculated based on MNE-Version 0.24.1 guidelines
            ### function: mne.minimum_norm.resolution_metrics
            # transpose resolution matrix for CTFs
            if function == 'ctf':
                resmat = resmat.T

            # Euclidean distance between true and peak locations
            if metric == 'peak_err':
                # peak locations (max)
                resmax = [abs(col).argmax() for col in resmat.T]
                maxloc = locations[resmax, :]
                # distance between true and peak locations
                diffloc = locations - maxloc
                locerr = np.linalg.norm(diffloc, axis=1)

            parcellatedData = np.zeros((len(parcelIDs), 1))
            for IDnum in parcelIDs:
                idx = np.where(parcels == IDnum)[0]
                parcellatedData[IDnum-1, :] = np.nanmean(locerr[idx],
                                                         axis=0)

            group_metric_vertex.append(locerr)
            group_metric_parcel.append(np.squeeze(parcellatedData))

            endTime = time.time()

            print('\nSubj %s, dataType %s, function %s' % (s, dataType,
                                                           function),
                  '\n Run time: %s' % (endTime - startTime))

        avg_vertex = np.mean(np.array(group_metric_vertex), axis=0)
        avg_parcel = np.mean(np.array(group_metric_parcel), axis=0)

        path_to_output = os.path.join(hcp_dir, 'brainstormResults',
                                      dataType)

        outputName_vertex = os.path.join(path_to_output, ('avg_' + metric + '_'
                                                          + function + '_' +
                                                          dataType +
                                                          '_vertex.npy'))
        np.save(outputName_vertex, avg_vertex)

        outputName_parcel = os.path.join(path_to_output, ('avg_' + metric + '_'
                                                          + function + '_' +
                                                          dataType +
                                                          '_Schaefer400.npy'))
        np.save(outputName_parcel, avg_parcel)
