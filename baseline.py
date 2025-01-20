# %% Importing libraries

import polars as pl
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, StratifiedKFold, StratifiedGroupKFold, GroupKFold
from sklearn.metrics import roc_auc_score, f1_score,matthews_corrcoef, precision_recall_curve, auc
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression, Lasso

from cir_model import CenteredIsotonicRegression
from lightgbm import LGBMRegressor, LGBMClassifier, log_evaluation, early_stopping
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor, XGBClassifier
from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
import optuna
import gc

import warnings
warnings.filterwarnings('ignore')

import random
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
seed_everything(seed=42)
