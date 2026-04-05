
import src.experiment.experiments as exp
from src.utils import preprocess, consts, data_objs

import numpy as np
import logging

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

def train():
    RAND_STATE_INT = 10
    rng = np.random.default_rng()

    # Load Data
    logging.info("--Loading data ...")
    sleep_subj_0 = data_objs.SleepSubject(subject=consts.get_train_subj(), epoch_duration=consts.get_epoch_duration_const())
    recording_data = sleep_subj_0.get_recording(recording_no=consts.get_train_recording())

    logging.info("--Bandpower from epochs ...")
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
    train_data = data_objs.Data(X=X_train, y=y_train)
    test_data = data_objs.Data(X=X_test, y=y_test)

    logging.info("--Starting training ...")

    # exp.exp_model_screening(train_data, test_data, RAND_STATE_INT)
    # exp.exp_model_hyperparameter_sweep(train_data, test_data, RAND_STATE_INT)
    exp.exp_model_finetuning(train_data, test_data, RAND_STATE_INT)

    # Package models

def main():
    logging.basicConfig(
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    train()

if __name__ == "__main__":
    main()