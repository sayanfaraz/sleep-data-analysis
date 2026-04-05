
import src.experiment.experiments as exp
import src.experiment.pipeline as pipeline
from src.utils import preprocess, consts, data_objs

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# import mne
# import pprint

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

def apply_smote(X, y):
    sm = SMOTE()
    return sm.fit_resample(X, y)

def train():
    RAND_STATE_INT = 10
    rng = np.random.default_rng()

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

    X_train, X_test, y_train, y_test = train_test_split(dataset_df[list(dataset_df.columns)[:-1]],
                                                        dataset_df[list(dataset_df.columns)[-1]],
                                                        test_size=0.33,
                                                        random_state=RAND_STATE_INT)
    
    # exp.exp_model_screening(X_train, X_test, y_train, y_test, RAND_STATE_INT)
    exp.exp_model_hyperparameter_sweep(X_train, y_train, RAND_STATE_INT)
    # exp.exp_model_finetuning(X_train, X_test, y_train, y_test, RAND_STATE_INT)

    # Package models


    # cm = confusion_matrix(y_test, y_pred)
    # mlflow.log_metric("cm", cm)

def main():
    # train()
    exp.load_best_hyperparams()

if __name__ == "__main__":
    train()


class SubjectDataManager():

    def __init__(self, subject: int) -> None:
        self._subject = subject
        self._train_recordings = []
        self._test_recordings = []

    def get_subject(self):
        return self._subject