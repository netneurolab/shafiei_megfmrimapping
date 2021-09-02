%% process HCP data with Brainstorm
hcp_dir = '/path/to/megdata/and/results/';
reports_dir = strcat(hcp_dir, 'brainstormReports/');

loadedsubj = load(fullfile(hcp_dir, 'myMEGList.mat'));
subjList = split(loadedsubj.myMEG, '_');
subjList = subjList(:,2);

channels = load(fullfile(hcp_dir, 'myMEGbadChannels'));
badChannels = channels.BadChannels;

% add path to Brainstorm
addpath(genpath('/usr/local/brainstorm3/'));

% Run Brainstorm
fcn_hcp_meg_process_connectivity(hcp_dir, subjList, badChannels, reports_dir)
