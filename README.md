# Mapping electromagnetic networks to haemodynamic networks in the human brain
This repository contains processing scripts and data in support of the preprint:

Shafiei, G., Baillet, S., & Misic, B. (2021). Mapping electromagnetic networks to haemodynamic networks in the human brain. bioRxiv.
[https://www.biorxiv.org/content/10.1101/2021.09.07.458941v1.abstract](https://www.biorxiv.org/content/10.1101/2021.09.07.458941v1.abstract)

## `code`
The [code](code/) folder contains all the code used to run the analyses and generate the figures.
All code in [preprocessing](code/preprocessing/) folder was written in Matlab and was used to preprocess MEG HCP data using [Brainstorm](https://neuroimage.usc.edu/brainstorm/Introduction).
All code in [analysis](code/analysis/) folder was written in Python and was used to analyze the preprocessed data.
I regularly use [netneurotools](https://github.com/netneurolab/netneurotools), a handy Python package developed in-house.

The [preprocessing](code/preprocessing/) folder contains the following files:
- [fcn_hcp_meg_process_connectivity.m](code/preprocessing/fcn_hcp_meg_process_connectivity.m) is the main function used to preprocess MEG HCP data. It relies on [Brainstorm](https://neuroimage.usc.edu/brainstorm/Introduction) and is a modified version of [Brainstorm tutorial](https://neuroimage.usc.edu/brainstorm/Tutorials/HCP-MEG) to preprocess resting-state MEG data from HCP.
- [scpt_runBrainstorm.m](code/preprocessing/scpt_runBrainstorm.m) is the script that runs [fcn_hcp_meg_process_connectivity.m](code/preprocessing/fcn_hcp_meg_process_connectivity.m).

The [analysis](code/analysis/) folder contains the following files:
- [fcn_megfmri.py](code/analysis/fcn_megfmri.py) contains all the functions used in main analysis ([scpt_megfmri.py](code/analysis/scpt_megfmri.py))
- [scpt_megfmri.py](code/analysis/scpt_megfmri.py) contains the script to run the main analyses and generate the figures of the manuscript.

## `data`
The [data](data/) folder contains the data used to run the analyses. Specifically, it containes the preprocessed, parcellated group-average MEG and fMRI functional connectivity matrices from 33 unrelated subjects in HCP. Note that HCP data redistribution must follow their [data terms](https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms). If you use any of the HCP data, please note that you must register with ConnectomeDB, agree to their terms and sign up for Open Access Data [here](https://www.humanconnectome.org/study/hcp-young-adult/data-use-terms). Please also cite relevant publications as mentioned [here](https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms).

The [data](data/) folder also contains required files to use and plot brain maps with [Schaefer atlas](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal).

