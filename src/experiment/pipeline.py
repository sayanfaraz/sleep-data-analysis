import lightgbm as lgb
import numpy as np
import sklearn.preprocessing
import optuna

from imblearn import FunctionSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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
        "RandomForest": RandomForestClassifier(n_jobs=-1),
        "SVM": SVC(probability=True),    #  change to False later, and proba -> decision_function
        "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(reg_param=0.2),  # need to tune
        "LightGBM": lgb.LGBMClassifier(objective='multiclass', verbosity=-1, n_jobs=-1)
        # Maybe try a neural net? cuz why not lol
        # Try an ensemble eventually
    }
    return models

def get_tuning_params(trial: optuna.Trial, model_name, X_train, y_train, RAND_STATE_INT):
    all_params = {
        "RandomForest": lambda: {
            'n_estimators': trial.suggest_int('n_estimators', low=100, high=1000, log=True),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', low=1, high=100),  # TODO: change to dependent on |X_train|
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'max_depth': trial.suggest_int("max_depth", 4, 40)
            
            # 'class_weight'
            # 'ccp_alpha': trial.suggest_categorical('ccp_alpha', calc_randforest_ccp_range(X_train, y_train, RAND_STATE_INT)), # Categorical because model only changes for values in ccp_alphas
        },
        "SVM": lambda: {
            "C": trial.suggest_float("C", low=1e-3, high=1e3, log=True),
            "gamma": trial.suggest_float("gamma", low=1e-5, high=1, log=True),
            # 'class_weight'

            # "degree": trial.suggest_int("degree", low=2, high=7)   #  im using rbf rn not poly - can fix later
        },
        "QuadraticDiscriminantAnalysis": lambda: {
            "reg_param": trial.suggest_float("reg_param", low=0.0, high=1.0),
        },
        "LightGBM": lambda: {
            "num_leaves": trial.suggest_int("num_leaves", low=20, high=300),
            "min_child_samples": trial.suggest_int("min_child_samples", low=5, high=200, log=True),
            
            "subsample": trial.suggest_float("subsample", low=0.5, high=1.),
            "subsample_freq": 1,
            "colsample_bytree": 1, # We only have 5 features lol

            "reg_alpha": trial.suggest_float("reg_alpha", low=1e-8, high=10., log=True), # L1
            "reg_lambda": trial.suggest_float("reg_lambda", low=1e-8, high=10., log=True) # L2

            # 'class_weight'
        }
    }
    return all_params[model_name]()

def calc_randforest_ccp_range(X_train, y_train, RAND_STATE_INT):
    tree = DecisionTreeClassifier(random_state=RAND_STATE_INT)
    ccp_path = tree.cost_complexity_pruning_path(X_train, y_train)

    ccp_alphas = ccp_path.ccp_alphas
    ccp_alphas = np.unique(ccp_alphas[ccp_alphas > 0]) # Unique alphas only, > 0

    return ccp_alphas

def get_scalers():
    scalers = {
        'default': 'passthrough',
        # 'log': sklearn.preprocessing.FunctionTransformer(lambda: np.log1p)
    }
    return scalers
