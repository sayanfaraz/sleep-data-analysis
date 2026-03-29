
from src import data_objs, consts, preprocess

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
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

models = {
    "RandomForest": RandomForestClassifier(),
    "SVM": SVC(probability=True)    #  change to False later, and proba -> decision_function
}

def apply_smote(X, y):
    sm = SMOTE()
    return sm.fit_resample(X, y)

def train():
    RAND_STATE_INT = 10
    rng = np.random.default_rng()

    mlflow.set_experiment("EEG Classification")

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

    for model_name, model in models.items():
        mlflow.sklearn.autolog()

        with mlflow.start_run(run_name=model_name):

            # Hyperparameter search
            

            # Train
            print("Train ...")
            
            # sm = SMOTE(random_state = RAND_STATE_INT)
            # X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)


            # rf = RandomForestClassifier()
            # rf.fit(X_train, y_train)

            rf = RandomForestClassifier()
            rf.fit(X_train, y_train)
            # rf.fit(X_train_smote, y_train_smote)

            # Val
            train_score = rf.score(X_train, y_train)
            
            y_pred = rf.predict(X_test)
            y_prob = rf.predict_proba(X_test)  # for AUC

            mlflow.log_metrics({
                'accuracy': accuracy_score(y_test, y_pred),
                "f1_weighted": f1_score(y_test, y_pred, average='weighted'),
                "f1_macro": f1_score(y_test, y_pred, average='macro'),
                "precision": precision_score(y_test, y_pred, average='weighted'),
                "recall": recall_score(y_test, y_pred, average='weighted'),
                "auc_weighted": roc_auc_score(y_test, y_prob, multi_class='ovo', average='weighted'),
                "auc_macro": roc_auc_score(y_test, y_prob, multi_class='ovo', average='macro')
            })

            
            # cm = confusion_matrix(y_test, y_pred)
            # accuracy = accuracy_score(y_test, y_pred)
            # precision = precision_score(y_test, y_pred, average='weighted')
            # recall = recall_score(y_test, y_pred, average='weighted')
            # f1 = f1_score(y_test, y_pred, average='weighted')

            # mlflow.log_metric("cm", cm)
            # mlflow.log_metric("accuracy", accuracy)
            # mlflow.log_metric("precision", precision)
            # mlflow.log_metric("recall", recall)
            # mlflow.log_metric("f1", f1)

        # Package models


if __name__ == "__main__":
    train()


class SubjectDataManager():

    def __init__(self, subject: int) -> None:
        self._subject = subject
        self._train_recordings = []
        self._test_recordings = []

    def get_subject(self):
        return self._subject