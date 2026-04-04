from imblearn import FunctionSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sklearn.preprocessing
import numpy as np
from itertools import product

def exclusions():
    return [
        {'model': 'RandomForest', 'sampler': 'SMOTE'},
        {'model': 'RandomForest', 'scaler': 'log'},
    ]

def all_configs_generator(rand_int):
    '''
        Iterator for each possible configuration, each iter yielding a dictionary as follows:
            {'model': __ModelClass__, 'sampler': __sampler()__, 'scaler': __scaler()__, ...}
    '''
    axes = {
        'model': get_models(),
        'sampler': get_samplers(rand_int),
        'scaler': get_scalers()
    }

    for combo in product(*axes.values()):
        yield dict(zip(axes.keys(), combo))

def get_samplers(rand_int):
    samplers = {
        'default': FunctionSampler(),
        'smote': SMOTE(random_state=rand_int)
    }
    return samplers

def make_datasets(samplers: dict, X_train, y_train):
    datasets = {}
    for sampler_name, sampler in samplers.items():
        datasets[sampler_name] = sampler.fit_resample(X_train, y_train)
    return datasets

# def config_to_pipeline(config, X_train, y_train):
#     return ImbPipeline(steps=[
#         ('scaler': config['scaler']),
#         ('model': )
#     ])

def get_models():
    models = {
        "RandomForest": RandomForestClassifier(),
        "SVM": SVC(probability=True),    #  change to False later, and proba -> decision_function
        "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(reg_param=0.2),  # need to tune
        "LightGBM": lgb.LGBMClassifier(objective='multiclass')
        # Maybe try a neural net? cuz why not lol
        # Try an ensemble eventually
    }
    return models

def get_models_old():
    models = {
        "RandomForest": {
            'model': RandomForestClassifier(),
            'try_smote': False,
            'try_logrel': False
        },
        "SVM": {
            'model': SVC(probability=True),    #  change to False later, and proba -> decision_function
            'try_smote': True,
            'try_logrel': True
        },
        "QuadraticDiscriminantAnalysis": {
            'model': QuadraticDiscriminantAnalysis(reg_param=0.2),  # need to tune
            'try_smote': True,
            'try_logrel': True
        },
        "LightGBM": {
            'model': lgb.LGBMClassifier(
                objective='multiclass'
            ),
            'try_smote': True,
            'try_logrel': True
        }
        # Maybe try a neural net? cuz why not lol
        # Try an ensemble eventually
    }
    return models


def get_scalers():
    scalers = {
        'default': 'passthrough',
        # 'log': sklearn.preprocessing.FunctionTransformer(lambda: np.log1p)
    }
    return scalers
