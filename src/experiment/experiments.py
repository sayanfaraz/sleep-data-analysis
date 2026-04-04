import src.experiment.evaluation as evaluation
import src.experiment.pipeline as exp_pipeline

import mlflow
from imblearn.over_sampling import SMOTE

def get_model_screening_exp_name():
    return "EEG_Classification-Model_Family_Screening"
    # return "Default_Name"

def get_model_screening_finetuning_name():
    return "EEG_Classification-Model_Finetuning"
    # return "Default_Name"

def top_n():
    return 3

def get_top_n_models(n):
    top_n_models = []
    
    # Pick Top N model families, and fine-tune them
    runs = mlflow.search_runs(
        experiment_names=[get_model_screening_exp_name()],
        order_by=["metrics.f1_macro DESC"]
    )

    for run_tuple in runs.head(n).iterrows():
        run = run_tuple[1]
        top_n_models.append(run)

    return top_n_models

def exp_model_finetuning(X_train, X_test, y_train, y_test, RAND_STATE_INT):
    top_n_models = get_top_n_models(top_n())

    datasets = exp_pipeline.make_datasets(exp_pipeline.get_samplers(RAND_STATE_INT), X_train, y_train)    

    mlflow.set_experiment(get_model_screening_finetuning_name())
    mlflow.sklearn.autolog()

    for model_metadata in top_n_models:
        model_name = model_metadata['tags.model_family']
        model = exp_pipeline.get_models()[model_name]

        sampler_name = model_metadata['tags.sampler']
        X_train_final, y_train_final = datasets[sampler_name]

        run_name=f"{model_name}_finetuned__{sampler_name if sampler_name!='default' else ''}"
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