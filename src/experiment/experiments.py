import src.experiment.evaluation as evaluation
import src.experiment.pipeline as exp_pipeline

import mlflow
import optuna
import json

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import cross_val_score

def get_model_screening_exp_name():
    return "EEG_Classification-Model_Family_Screening"
    # return "Default_Name"

def get_model_hyperparameters_exp_name():
    return "EEG_Classification-Model_Hyperparameter_Sweep"

def get_model_finetuning_exp_name():
    return "EEG_Classification-Model_Finetuning"
    # return "Default_Name"

def get_best_params_filename():
    return "best_params.json"

def top_n():
    return 3

def load_top_n_models(n):
    top_n_models = []
    
    # Pick Top N model families, and fine-tune them
    runs = mlflow.search_runs(
        experiment_names=[get_model_screening_exp_name()],
        order_by=["metrics.f1_macro DESC"]
    )

    for i, run in runs.head(n).iterrows():
        top_n_models.append(run)

    return top_n_models

def load_best_hyperparams():
    best_params_per_model = []
    runs = mlflow.search_runs(
        experiment_names=[get_model_hyperparameters_exp_name()]
    )
    client = mlflow.tracking.MlflowClient()

    for i, run in runs.iterrows():
        best_params_f = client.download_artifacts(run['run_id'], get_best_params_filename())
        model_family = run['tags.model_family']
        with open(best_params_f) as f:
            best_params = json.load(f)

            metadata = run.to_dict()
            metadata["best_params"] = best_params

            best_params_per_model.append(metadata)

            print(model_family)
            print(best_params)
            print("\n\n")

    return best_params_per_model

def make_objective(model_name, X_train, y_train, RAND_STATE_INT):
    def objective(trial):
        param_sweep = exp_pipeline.get_tuning_params(trial, X_train, y_train, RAND_STATE_INT)[model_name]
        model = exp_pipeline.get_models()[model_name]
        model.set_params(**param_sweep)

        score = cross_val_score(model, X_train, y_train, cv=2, scoring="f1_macro").mean()   # TODO: change cv to at least 10
        return score
        
    return objective

def exp_model_hyperparameter_sweep(X_train, y_train, RAND_STATE_INT):
    optuna.logging.set_verbosity(optuna.logging.WARNING) 
    
    top_n_models = load_top_n_models(top_n())

    datasets = exp_pipeline.make_datasets(exp_pipeline.get_samplers(RAND_STATE_INT), X_train, y_train)    

    mlflow.set_experiment(get_model_hyperparameters_exp_name())
    for model_metadata in top_n_models:
        model_name = model_metadata['tags.model_family']

        sampler_name = model_metadata['tags.sampler']
        X_train_final, y_train_final = datasets[sampler_name]

        run_name=f"{model_name}_hyperparam_sweep__{sampler_name if sampler_name!='default' else ''}"
        print("\n\n\n" + run_name + ", f1_macro: " + str(model_metadata['metrics.f1_macro']))

        study = optuna.create_study(direction='maximize')
        objective = make_objective(model_name, X_train_final, y_train_final, RAND_STATE_INT)
        study.optimize(objective, n_trials=5)  # should be 100

        # 4. Results
        print("Best params:", study.best_params)
        print("Best score: ", study.best_value)

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                "stage": "hyperparam_sweep",
                "sampler": sampler_name,
                "model_family": model_name
            })

            mlflow.log_dict(study.best_params, get_best_params_filename())
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_f1score", study.best_value)
        
def exp_model_finetuning(X_train, X_test, y_train, y_test, RAND_STATE_INT):
    top_n_models = load_top_n_models(top_n())

    datasets = exp_pipeline.make_datasets(exp_pipeline.get_samplers(RAND_STATE_INT), X_train, y_train)

    mlflow.set_experiment(get_model_finetuning_exp_name())
    mlflow.sklearn.autolog()

    for model_metadata in top_n_models:
        model_name = model_metadata['tags.model_family']
        
        sampler_name = model_metadata['tags.sampler']
        X_train_final, y_train_final = datasets[sampler_name]

        run_name=f"{model_name}_hyperparam_sweep__{sampler_name if sampler_name!='default' else ''}"
        print("\n\n\n" + run_name + ", f1_macro: " + str(model_metadata['metrics.f1_macro']))

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                "stage": "finetuning",
                "sampler": sampler_name,
                "model_family": model_name
            })

            model.fit(X_train_final, y_train_final)
            train_score = model.score(X_train_final, y_train_final)

            # Val
            evaluation.evaluate(model, X_test, y_test)

def exp_model_screening(X_train, X_test, y_train, y_test, RAND_STATE_INT):
    datasets = exp_pipeline.make_datasets(exp_pipeline.get_samplers(RAND_STATE_INT), X_train, y_train)

    mlflow.set_experiment(get_model_screening_exp_name())

    for config in exp_pipeline.all_configs_generator(RAND_STATE_INT):
        model_name = config['model']
        model = exp_pipeline.get_models()[model_name]

        sampler_name = config['sampler']

        run_name=f"{model_name}_{sampler_name if sampler_name!='default' else ''}"
        print("\n\n\n" + run_name)

        X_train_final, y_train_final = datasets[sampler_name]

        with mlflow.start_run(run_name=run_name):
            mlflow.sklearn.autolog()

            mlflow.set_tags({
                "stage": "screening",
                "sampler": sampler_name,
                "model_family": model_name
            })

            model.fit(X_train_final, y_train_final)
            train_score = model.score(X_train_final, y_train_final)

            # Val
            evaluation.evaluate(model, X_test, y_test)