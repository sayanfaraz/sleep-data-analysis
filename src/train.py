
from src import data_objs, consts, preprocess

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import mne
# import pprint
import mlflow

# def train(subject, recording, epoch_duration):
#     load_data

#     preprocess_data

#     train

def train():
    RAND_STATE_INT = 10
    rng = np.random.default_rng()

    mlflow.sklearn.autolog()

    # Load Data
    print("Loading data ...")
    sleep_subj_0 = data_objs.SleepSubject(subject=consts.get_train_subj(), epoch_duration=consts.get_epoch_duration_const())
    recording_data = sleep_subj_0.get_recording(recording_no=consts.get_train_recording())

    print("Bandpower from epochs ...")
    sleep_stage_rel_bandpower, sleep_stage_abs_bandpower = preprocess.bandpowers_from_epochs(
        recording_data.epochs,
        recording_data.raw,
        consts.get_event_ids(),
        recording_data.sfreq,
        channel='EEG Fpz-Cz'
    )

    dataset_df = preprocess.bandpower_dict_to_df(sleep_stage_rel_bandpower, consts.get_event_ids())

    # Hyperparameter search
    

    # Train
    print("Train ...")
    X_train, X_test, y_train, y_test = train_test_split(dataset_df[list(dataset_df.columns)[:-1]],
                                                  dataset_df[list(dataset_df.columns)[-1]],
                                                  test_size=0.33,
                                                  random_state=RAND_STATE_INT)


    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    # Val
    # y_pred = rf.predict(X_test)

    rf.score(X_test, y_test)

    # cm = confusion_matrix(y_test, y_pred)
    # accuracy = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred, average='weighted')
    # recall = recall_score(y_test, y_pred, average='weighted')
    # f1 = f1_score(y_test, y_pred, average='weighted')

    # Package models


if __name__ == "__main__":
    main()


class SubjectDataManager():

    def __init__(self, subject: int) -> None:
        self._subject = subject
        self._train_recordings = []
        self._test_recordings = []

    def get_subject(self):
        return self._subject