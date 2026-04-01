import src.experiment.evaluation as evaluation

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

def exp_model_finetuning(models, X_train, X_test, y_train, y_test, RAND_STATE_INT):
    top_n_models = get_top_n_models(top_n())

    sm = SMOTE(random_state = RAND_STATE_INT)
    X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)

    mlflow.set_experiment(get_model_screening_finetuning_name())
    mlflow.sklearn.autolog()

    for model_metadata in top_n_models:
        # print(run['tags.model_family'], , model_metadata['metrics.f1_macro'])

        model_name = model_metadata['tags.model_family']
        model = models[model_name]['model']
        smote_on = model_metadata['tags.smote'] == 'True'

        X_train_final = X_train if smote_on==False else X_train_smote
        y_train_final = y_train if smote_on==False else y_train_smote

        run_name=f"{model_name}_finetuned{'__SMOTE' if smote_on else ''}"

        with mlflow.start_run(run_name=run_name):

            mlflow.set_tags({
                "stage": "finetuning",
                "smote": smote_on,
                "model_family": model_name
            })

            model.fit(X_train_final, y_train_final)
            train_score = model.score(X_train_final, y_train_final)

            # Val
            evaluation.evaluate(model, X_test, y_test)
        

def exp_model_screening(models, X_train, X_test, y_train, y_test, RAND_STATE_INT):
    sm = SMOTE(random_state = RAND_STATE_INT)
    X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)

    mlflow.set_experiment(get_model_screening_exp_name())
    mlflow.sklearn.autolog()

    for model_name, model_attrs in models.items():
        model = model_attrs['model']
        try_smote = model_attrs['try_smote']
        try_logrel = model_attrs['try_logrel']

        for smote_on in [False, True]:
            if try_smote==False and smote_on==True:
                continue

            X_train_final = X_train if smote_on==False else X_train_smote
            y_train_final = y_train if smote_on==False else y_train_smote

            run_name=f"{model_name}{'__SMOTE' if smote_on else ''}"

            with mlflow.start_run(run_name=run_name):

                mlflow.set_tags({
                    "stage": "screening",
                    "smote": smote_on,
                    "model_family": model_name
                })

                model.fit(X_train_final, y_train_final)
                train_score = model.score(X_train_final, y_train_final)

                # Val
                evaluation.evaluate(model, X_test, y_test)