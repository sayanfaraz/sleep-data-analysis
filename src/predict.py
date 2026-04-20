import logging

import mlflow
import numpy as np
from sklearn.model_selection import train_test_split

import src.experiment.evaluation as exp_eval
from src.utils import data_objs, preprocess, consts

def predict():
    champion_model = mlflow.sklearn.load_model(f"models:/{consts.get_champion_model_name()}@champion")

    print(champion_model)

    RAND_STATE_INT = 10
    rng = np.random.default_rng()

    # Load Data
    logging.info("--Loading data ...")
    sleep_subj_0 = data_objs.SleepSubject(subject=consts.get_train_subj(), epoch_duration=consts.get_epoch_duration_const())
    recording_data = sleep_subj_0.get_recording(recording_no=consts.get_predict_recording())
    # recording_data = sleep_subj_0.get_recording(recording_no=consts.get_train_recording())

    logging.info("--Bandpower from epochs ...")
    sleep_stage_rel_bandpower, sleep_stage_abs_bandpower = preprocess.bandpowers_from_epochs(
        recording_data.epochs,
        recording_data.raw,
        consts.get_event_ids(),
        recording_data.sfreq,
        channel='EEG Fpz-Cz'
    )

    dataset_df = preprocess.bandpower_dict_to_df(sleep_stage_rel_bandpower, consts.get_event_ids())

    X_train, X_pred, y_train, y_pred = train_test_split(dataset_df[list(dataset_df.columns)[:-1]],
                                                        dataset_df[list(dataset_df.columns)[-1]],
                                                        test_size=0.999,
                                                        random_state=RAND_STATE_INT)
    print(X_pred.shape)
    print(X_train.shape)
    pred_data = data_objs.Data(X=X_pred, y=y_pred)

    logging.info("--Starting prediction ...")

    print(champion_model)
    print(champion_model.predict(X_pred))

    metrics = exp_eval.evaluate(champion_model, X_pred, y_pred)

    for metric, val in metrics.items():
        print(f"{metric}: {val}")