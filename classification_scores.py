
from pyriemann.classification import MDM
from pyriemann.estimation import ERPCovariances
from tqdm import tqdm

import sys
sys.path.append('.')
from braininvaders2014b.dataset import BrainInvaders2014b

from scipy.io import loadmat
import numpy as np
import mne

from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
 
dataset = BrainInvaders2014b()

scores = {}
for pair in dataset.pair_list:
    scores[pair] = {}

    print('pair', str(pair))

    sessions = dataset._get_single_pair_data(pair=pair)

    for subject in [1, 2]:
        scores[pair][subject] = {}

        print('subject', subject)

        # subject 1
        raw_solo = sessions['solo_' + str(subject)]['run_1']        
        if subject == 1:
            pick_channels = raw_solo.ch_names[0:32] + [raw_solo.ch_names[-1]]
        elif subject == 2:
            pick_channels = raw_solo.ch_names[32:-1] + [raw_solo.ch_names[-1]]        
        raw_solo.pick_channels(pick_channels)
        raw_cola = sessions['collaborative']['run_1']
        raw_cola = raw_cola.copy().pick_channels(pick_channels)

        for condition, raw in zip(['solo', 'cola'], [raw_solo, raw_cola]):        

            # filter data and resample
            fmin = 1
            fmax = 20
            raw.filter(fmin, fmax, verbose=False)            

            # detect the events and cut the signal into epochs
            events = mne.find_events(raw=raw, shortest_event=1, verbose=False)
            event_id = {'NonTarget': 1, 'Target': 2}
            epochs = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=0.8, baseline=None, verbose=False, preload=True)
            epochs.pick_types(eeg=True)

            # get trials and labels
            X = epochs.get_data()
            y = epochs.events[:,-1]
            y = y - 1

            # cross validation
            skf = StratifiedKFold(n_splits=5)
            clf = make_pipeline(ERPCovariances(estimator='lwf', classes=[1]), MDM())
            scr = cross_val_score(clf, X, y, cv=skf, scoring = 'roc_auc').mean()
            scores[pair][subject][condition] = scr

            print(condition, scr)

    print('')

filename = 'classification_scores.pkl'
joblib.dump(scores, filename)    

with open('classification_scores.txt', 'w') as the_file:
    for pair in scores.keys():
        the_file.write('pair ' + str(pair).zfill(2) + ', subject 1 (solo) : ' + '{:.2f}'.format(scores[pair][1]['solo']) + '\n')
        the_file.write('pair ' + str(pair).zfill(2) + ', subject 1 (cola) : ' + '{:.2f}'.format(scores[pair][1]['cola']) + '\n')
        the_file.write('pair ' + str(pair).zfill(2) + ', subject 2 (solo) : ' + '{:.2f}'.format(scores[pair][2]['solo']) + '\n')
        the_file.write('pair ' + str(pair).zfill(2) + ', subject 2 (cola) : ' + '{:.2f}'.format(scores[pair][2]['cola']) + '\n')
