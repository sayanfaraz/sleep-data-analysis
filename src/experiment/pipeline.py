from imblearn import FunctionSampler
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def get_models():
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

def get_samplers(rand_int):
    samplers = {
        'default': FunctionSampler(),
        'smote': SMOTE(random_state=rand_int)
    }
    return samplers


def get_scalers():
    scalers = {
        'none': 'passthrough',
        'log': 'passthrough'
    }