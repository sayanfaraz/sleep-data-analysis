
import mlflow
import pandas as pd

import src.experiment.experiments as exp
from src.utils import preprocess, consts, data_objs

import numpy as np
import logging

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

from src.utils.consts import get_champion_model_name

# TODO: Turn sklearn autologs into manual logs
# TODO: Save as skops or ONYXX instead of pickle
# TODO: Register models, metadata, and overall pipeline
# TODO: Load model and pipeline, and run on inference

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
    # exp.exp_model_finetuning(train_data, test_data, RAND_STATE_INT)

    # Package models
    logging.info("--Packaging best model ...")
    update_model_registry()

def update_model_registry():
    client = mlflow.MlflowClient()

    challenger_metadata = exp.load_top_n_models(3, experiment_name=exp.get_model_finetuning_exp_name(),
                                                orderby_metric="metrics.f1_weighted")[0]
    # print(challenger_metadata)
    # print("/n")
    # for meta in challenger_metadata:
    #     print(meta)
    #     print("\n__________________________\n")
    
    # for row in challenger_metadata.items():
    #     print(row)

    run_id = challenger_metadata['run_id'] # type: ignore
    # # run_id = "9d41d77c74da4124801db9f29cbf2542"
    # # print(run_id)
    # artifacts = client.list_artifacts(run_id, "")
    # for a in artifacts:
    #     print(a.path, a.is_dir)

    uri = f"runs:/{run_id}/model"

    # # print(uri)
    # # print(repr(uri))

    # loaded = mlflow.sklearn.load_model(uri)
    # print(loaded)  # if this works, the artifact is fine
    logging.info("Registering Model ...")
    registered = mlflow.register_model(model_uri=uri, name=get_champion_model_name())
    copy_tags_to_registered_model(registered, challenger_metadata, client)

    # # champion_metadata = 

    # # if challenger_is_better(champion_metadata, challenger_metadata):
    client.set_registered_model_alias(get_champion_model_name(), "champion", registered.version)

def challenger_is_better(champion_metadata, challenger_metadata):
    return challenger_metadata[exp.scoring_metric_ind()] > champion_metadata[exp.scoring_metric_ind()]

def copy_tags_to_registered_model(
    registered_modelv: mlflow.entities.model_registry.ModelVersion,
    challenger_metadata,
    client: mlflow.MlflowClient
):
    custom_tags = {
        k.replace("tags.", ""): v
        for k, v in challenger_metadata.items()
        if k.startswith("tags.") and not k.startswith("tags.mlflow.")
    }

    metric_tags = {
        k.replace("metrics.", ""): str(v)
        for k, v in challenger_metadata.items()
        if k.startswith("metrics.")
    }

    for key, value in {**custom_tags, **metric_tags}.items():
        client.set_model_version_tag(
            registered_modelv.name,
            registered_modelv.version,
            key,
            value
        )

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