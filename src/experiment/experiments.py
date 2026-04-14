import src.experiment.evaluation as exp_evaluation
import src.experiment.pipeline as exp_pipeline

from src.utils import data_objs as dobjs

import logging
import numpy as np
import mlflow
import optuna
import json
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score

def get_model_screening_exp_name():
    return "EEG_Classification-Model_Family_Screening"

def get_model_hyperparameters_exp_name():
    return "EEG_Classification-Model_Hyperparameter_Sweep"

def get_model_finetuning_exp_name():
    return "EEG_Classification-Model_Finetuning"

def get_scoring_metric_name():
    return 'f1_macro'

def scoring_metric_ind(metric_name=get_scoring_metric_name()):
    return 'metrics.' + metric_name

def get_best_params_filename():
    return "best_params.json"

def top_n():
    return 6   # So I can try out the other model families as well

def get_hyp_sweep_ntrials():
    return 50

def get_hyp_sweep_kfolds():
    return 5

def exp_model_hyperparameter_sweep(dtrain: dobjs.Data, RAND_STATE_INT):
    optuna.logging.set_verbosity(optuna.logging.WARNING) 
    
    top_n_models = load_top_n_models(top_n(), experiment_name=get_model_screening_exp_name())

    # TODO: put imbalanced-resampling into objective so its inside CV, not outside -> X_val will only have real values
    # TODO: to do ^, use an ImbPipeline (SMOTE -> model)
    datasets = exp_pipeline.make_datasets(exp_pipeline.get_samplers(RAND_STATE_INT), dtrain)    

    mlflow.set_experiment(get_model_hyperparameters_exp_name())
    logging.info(f"Starting Experiment: {get_model_hyperparameters_exp_name()}")
    for model_metadata in top_n_models:
        pipeline_attrs = exp_pipeline.PipelineAttrs(model_name=model_metadata['tags.model_family'],
                                                    sampler_name=model_metadata['tags.sampler'])

        train_final: dobjs.Data = datasets[pipeline_attrs.sampler_name]

        run_name = make_run_name(pipeline_attrs, exp_type="hyperparam_sweep")
        logging.info(f"\n{run_name}, {get_scoring_metric_name()}: {str(model_metadata[scoring_metric_ind()])}")

        study = optuna.create_study(direction='maximize')
        objective = make_objective(pipeline_attrs.model_name, train_final, RAND_STATE_INT,
                                   scoring=get_scoring_metric_name(),
                                   kfolds=get_hyp_sweep_kfolds()) # should be 10
        study.optimize(objective, n_trials=get_hyp_sweep_ntrials(), n_jobs=-1)  # should be 100

        # 4. Results
        logging.info(f"Best params: {study.best_params}")
        logging.info(f"Best score: {study.best_value}")

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(make_mlflow_tags(
                pipeline_attrs,
                stage="hyperparam_sweep")
            )

            mlflow.log_dict(study.best_params, get_best_params_filename())
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_f1score", study.best_value)

            if pipeline_attrs.model_name=="LightGBM":
                best_trial = study.best_trial
                mlflow.log_param("refit_n_estimators", best_trial.user_attrs["mean_best_iteration"])

def exp_model_finetuning(dtrain: dobjs.Data, dtest: dobjs.Data, RAND_STATE_INT):
    top_models_metadata = load_best_hyperparams()

    datasets = exp_pipeline.make_datasets(exp_pipeline.get_samplers(RAND_STATE_INT), dtrain)

    mlflow.set_experiment(get_model_finetuning_exp_name())
    logging.info(f"Starting Experiment: {get_model_finetuning_exp_name()}")
    mlflow.sklearn.autolog()  # pyright: ignore[reportPrivateImportUsage]

    for model_metadata in top_models_metadata:                                                   
        pipeline_attrs, model = model_from_hyperparam_metadata(model_metadata)

        train_final: dobjs.Data = datasets[pipeline_attrs.sampler_name]

        run_name = make_run_name(pipeline_attrs, exp_type="finetune")
        logging.info(f"\n{run_name}")

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(make_mlflow_tags(
                pipeline_attrs,
                stage="finetuning")
            )

            train_and_eval(model, train_final, dtest)

def exp_model_screening(dtrain: dobjs.Data, dtest: dobjs.Data, RAND_STATE_INT):
    datasets = exp_pipeline.make_datasets(exp_pipeline.get_samplers(RAND_STATE_INT), dtrain)

    mlflow.set_experiment(get_model_screening_exp_name())
    logging.info(f"Starting Experiment: {get_model_screening_exp_name()}")

    for config in exp_pipeline.all_configs_generator(RAND_STATE_INT):
        pipeline_attrs = exp_pipeline.PipelineAttrs(config['model'], config['sampler'])

        model = exp_pipeline.get_models()[pipeline_attrs.model_name]
        train_final: dobjs.Data = datasets[pipeline_attrs.sampler_name]

        run_name = make_run_name(pipeline_attrs, exp_type="")
        logging.info(f"\n{run_name}")

        with mlflow.start_run(run_name=run_name):
            mlflow.sklearn.autolog() # pyright: ignore[reportPrivateImportUsage]

            mlflow.set_tags(make_mlflow_tags(
                pipeline_attrs,
                stage="screening")
            )

            train_and_eval(model, train_final, dtest)

def load_top_n_models(n: int, experiment_name: str, orderby_metric=scoring_metric_ind()):
    top_n_models = []

    # Pick Top N model families, and fine-tune them
    orderby = orderby_metric + " DESC"
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=[orderby]
    )

    for i, run in runs.head(n).iterrows(): # type: ignore - We know its a pd.DataFrame based on mlflow.search_runs
        top_n_models.append(run)

    return top_n_models

def load_best_hyperparams():
    best_params_per_model = []
    runs = mlflow.search_runs(
        experiment_names=[get_model_hyperparameters_exp_name()]
    )
    client = mlflow.tracking.MlflowClient() # pyright: ignore[reportPrivateImportUsage]

    for i, run in runs.iterrows(): # type: ignore - We know its a pd.DataFrame based on mlflow.search_runs
        best_params_f = client.download_artifacts(run['run_id'], get_best_params_filename())
        model_family = run['tags.model_family']
        with open(best_params_f) as f:
            best_params = json.load(f)

            metadata = run.to_dict()
            metadata["best_params"] = best_params

            best_params_per_model.append(metadata)

    return best_params_per_model

def lgb_objective(trial, model_name, dtrain: dobjs.Data, n_splits, rand_state):
    params = exp_pipeline.get_tuning_params(trial, "LightGBM", dtrain, rand_state)
    params.update({
        "learning_rate": 0.05,      # fixed
        "n_estimators":  2000,      # high ceiling — early stopping will cut this down
        "verbosity":     -1,
        "random_state":  rand_state,
    })

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rand_state)
    fold_scores = []
    best_iterations = []

    for train_idx, val_idx in cv.split(dtrain.X, dtrain.y):
        X_fold_train, X_val = dtrain.X.iloc[train_idx], dtrain.X.iloc[val_idx] # type: ignore
        y_fold_train, y_val = dtrain.y.iloc[train_idx], dtrain.y.iloc[val_idx] # type: ignore

        model = exp_pipeline.get_models()[model_name]
        model.set_params(**params)

        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=-1),   # silences per-iteration output
            ],
        )

        best_iterations.append(model.best_iteration_)
        preds = model.predict(X_val)
        fold_scores.append(f1_score(y_val, preds, average='macro'))

    mean_score = np.mean(fold_scores)

    # Log the mean best iteration so you know what n_estimators to use at refit
    trial.set_user_attr("mean_best_iteration", int(np.mean(best_iterations)))

    return mean_score

def make_objective(model_name, dtrain: dobjs.Data, RAND_STATE_INT, scoring, kfolds):
    def objective(trial):
        param_sweep = exp_pipeline.get_tuning_params(trial, model_name, dtrain, RAND_STATE_INT)

        model = exp_pipeline.get_models()[model_name]
        model.set_params(**param_sweep)

        cv = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=RAND_STATE_INT) #TODO: Use TimeSeriesSplit to avoid temporal correlations
        score = cross_val_score(model, dtrain.X, dtrain.y, cv=cv, scoring=scoring, n_jobs=-1).mean()   # TODO: change cv to at least 10
        return score
    
    def lgb_obj(trial):
        return lgb_objective(trial, model_name, dtrain, n_splits=kfolds, rand_state=RAND_STATE_INT)

    return lgb_obj if model_name=="LightGBM" else objective

def model_from_hyperparam_metadata(model_metadata):
    pipeline_attrs = exp_pipeline.PipelineAttrs(model_name=model_metadata['tags.model_family'],
                                                sampler_name=model_metadata['tags.sampler'])
    params = model_metadata['best_params']

    model = exp_pipeline.get_models()[pipeline_attrs.model_name]
    model.set_params(**params)

    if pipeline_attrs.model_name=="LightGBM":
        refit_n_estimators = int(model_metadata["params.refit_n_estimators"])
        model.set_params(**{
            "learning_rate": 0.05,
            "n_estimators": refit_n_estimators,
            "verbosity": -1
        })

    return pipeline_attrs, model

def train_and_eval(model, dtrain, dtest):
    model.fit(dtrain.X, dtrain.y)
    train_score = model.score(dtrain.X, dtrain.y)

    exp_evaluation.evaluate(model, dtest.X, dtest.y)

    # cm = confusion_matrix(y_test, y_pred)
    # mlflow.log_metric("cm", cm)

def make_mlflow_tags(pipeline_attrs, stage) -> dict:

    return {
                "stage": stage,
                "sampler": pipeline_attrs.sampler_name,
                "model_family": pipeline_attrs.model_name
    }

def make_run_name(pipeline_attrs: exp_pipeline.PipelineAttrs, exp_type: str="") -> str:
    model_name = pipeline_attrs.model_name
    sampler_name = pipeline_attrs.sampler_name

    run_name=f"{model_name}_{exp_type}{"__" if exp_type!="" else ""}{sampler_name if sampler_name!='default' else ''}"

    return run_name