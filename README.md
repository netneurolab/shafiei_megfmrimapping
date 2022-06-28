# Human electromagnetic and haemodynamic networks systematically converge in unimodal cortex and diverge in transmodal cortex
This repository contains processing scripts and data in support of the preprint:

Shafiei, G., Baillet, S., & Misic, B. (2022). Human electromagnetic and haemodynamic networks systematically converge in unimodal cortex and diverge in transmodal cortex. bioRxiv.
[https://doi.org/10.1101/2021.09.07.458941](https://doi.org/10.1101/2021.09.07.458941)

## `code`
The [code](code/) folder contains all the code used to run the analyses and generate the figures.
All code in [preprocessing](code/preprocessing/) folder was written in Matlab and was used to preprocess MEG HCP data using [Brainstorm](https://neuroimage.usc.edu/brainstorm/Introduction).
All code in [analysis](code/analysis/) folder was written in Python and was used to analyze the preprocessed data.
I regularly use [netneurotools](https://github.com/netneurolab/netneurotools), a handy Python package developed in-house.

The [preprocessing](code/preprocessing/) folder contains the following files:
- [fcn_hcp_meg_process_connectivity.m](code/preprocessing/fcn_hcp_meg_process_connectivity.m) is the main function used to preprocess MEG HCP data. It relies on [Brainstorm](https://neuroimage.usc.edu/brainstorm/Introduction) and is a modified version of [Brainstorm tutorial](https://neuroimage.usc.edu/brainstorm/Tutorials/HCP-MEG) to preprocess resting-state MEG data from HCP.
- [scpt_runBrainstorm.m](code/preprocessing/scpt_runBrainstorm.m) is the script that runs [fcn_hcp_meg_process_connectivity.m](code/preprocessing/fcn_hcp_meg_process_connectivity.m).
- [resolution_metric.py](code/preprocessing/resolution_metric.py) is the script that is used to estimate MEG source localization error.

The [analysis](code/analysis/) folder contains the following files:
- [fcn_megfmri.py](code/analysis/fcn_megfmri.py) contains all the functions used in main analysis ([scpt_megfmri.py](code/analysis/scpt_megfmri.py))
- [scpt_megfmri.py](code/analysis/scpt_megfmri.py) contains the script to run the main analyses and generate the figures of the manuscript.
- [scpt_Fig1.py](code/analysis/scpt_Fig1.py), [scpt_Fig2.py](code/analysis/scpt_Fig2.py), and so on contain the scripts to regenerate the main and supplementary figures of the manuscript.
- [scpt_prepare_plotting_main.py](code/analysis/scpt_prepare_plotting_main.py) and [scpt_prepare_plotting_suppl.py](code/analysis/scpt_prepare_plotting_suppl.py) contain the scripts that were used to prepare summary data to plot the main and supplementary figures (available in [figures_data](data/figures_data/)).

## `data`
The [data](data/) folder contains the data used to run the analyses. Specifically, it containes the preprocessed, parcellated group-average MEG and fMRI functional connectivity matrices from 33 unrelated subjects in HCP. Note that HCP data redistribution must follow their [data terms](https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms). If you use any of the HCP data, please note that you must register with ConnectomeDB, agree to their terms and sign up for Open Access Data [here](https://www.humanconnectome.org/study/hcp-young-adult/data-use-terms). Please also cite relevant publications as mentioned [here](https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms).

The [figures_data](data/figures_data/) folder contains the summary data that can be directly used to regenerate the figures.

The [data](data/) folder also contains required files to use and plot brain maps with [Schaefer atlas](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal).

