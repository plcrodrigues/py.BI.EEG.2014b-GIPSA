#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import mne
import numpy as np
from . import download as dl
import os
import glob
import zipfile
import yaml
from scipy.io import loadmat
from distutils.dir_util import copy_tree
import shutil
import pandas as pd

BI2014b_URL = 'https://zenodo.org/record/3267302/files/'

class BrainInvaders2014b():
    '''
    This dataset contains electroencephalographic (EEG) recordings of 38 subjects 
    playing in pairs to the multi-user version of a visual P300-based Brain-Computer 
    Interface (BCI) named Brain Invaders (Congedo et al., 2011). The interface uses 
    the oddball paradigm on a grid of 36 symbols (1 Target, 35 Non-Target) that are 
    flashed pseudo-randomly to elicit a P300 response, an evoked-potential appearing 
    about 300ms after stimulation onset. EEG data were recorded using 32 active wet 
    electrodes per subjects (total: 64 electrodes) during three randomized conditions 
    (Solo1, Solo2, Collaboration). The experiment took place at GIPSA-lab, Grenoble, 
    France, in 2014. A full description of the experiment is available at 
    https://hal.archives-ouvertes.fr/hal-02173958. Python code for manipulating the 
    data is available at https://github.com/plcrodrigues/py.BI.EEG.2014b-GIPSA. 
    The ID of this dataset is bi2014b.
    '''

    def __init__(self):
        self.subject_list = list(range(1, 19 + 1))
        self.pair_list = self.subject_list

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        chnames = ['FP1','FP2','AFz','F7','F3','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8','CP5','CP1','CP2','CP6','P7','P3','Pz','P4','P8','PO7','O1','Oz','O2','PO8','PO9','PO10']
        chnames_subject1 = [chi + '_1' for chi in chnames]                                                      
        chnames_subject2 = [chi + '_2' for chi in chnames]      
        chnames = chnames_subject1 + chnames_subject2 + ['STI 014']
        chtypes = ['eeg'] * 64 + ['stim']               
        sessions = {}
        file_path_list = self.data_path(subject)        
        session_name_list = ['solo_1', 'solo_2', 'collaborative']
        for file_path, session_name in zip(file_path_list, session_name_list):

            sessions[session_name] = {}
            run_name = 'run_1'

            D = loadmat(file_path)['samples'].T

            S = D[1:65,:]
            stim = D[-1,:]
            X = np.concatenate([S, stim[None,:]])

            info = mne.create_info(ch_names=chnames, sfreq=512,
                                   ch_types=chtypes, montage=None,
                                   verbose=False)
            raw = mne.io.RawArray(data=X, info=info, verbose=False)

            sessions[session_name][run_name] = raw

        return sessions

    # dummy function just for more readable code
    def _get_single_pair_data(self, pair):
        return self._get_single_subject_data(pair)          

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):

        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))

        # check if has the .zip
        url = BI2014b_URL + 'group_' + str(subject).zfill(2) + '_mat.zip'
        path_zip = dl.data_path(url, 'BRAININVADERS2014B')
        path_folder = path_zip.strip('group_' + str(subject).zfill(2) + '_mat.zip')

        # check if has to unzip
        path_folder_subject = path_folder + 'group_' + str(subject).zfill(2) + os.sep
        if not(os.path.isdir(path_folder_subject)):
            os.mkdir(path_folder_subject)
            print('unzip', path_zip)
            zip_ref = zipfile.ZipFile(path_zip, "r")
            zip_ref.extractall(path_folder_subject)

        subject_paths = []

        # filter the data regarding the experimental conditions
        subject_paths.append(path_folder_subject + 'group_' + str(subject).zfill(2) + '_sujet_01.mat')        
        subject_paths.append(path_folder_subject + 'group_' + str(subject).zfill(2) + '_sujet_02.mat')        
        subject_paths.append(path_folder_subject + 'group_' + str(subject).zfill(2) + '.mat')        

        return subject_paths
