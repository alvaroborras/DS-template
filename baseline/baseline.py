# %% Importing libraries

import pandas as pd
import polars as pl
import numpy as np
from tqdm import tqdm
from loguru import logger

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
	KFold,
	StratifiedKFold,
	StratifiedGroupKFold,
	GroupKFold,
)
from sklearn.metrics import (
	roc_auc_score,
	f1_score,
	matthews_corrcoef,
	precision_recall_curve,
	auc,
	accuracy_score,
	precision_score,
	recall_score,
	mean_absolute_error,
	r2_score,
	mean_squared_error,
	mean_absolute_percentage_error,
	mean_squared_log_error,
)
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression, Lasso

from lightgbm import LGBMRegressor, LGBMClassifier, log_evaluation, early_stopping
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor, XGBClassifier
from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier

from typing import Dict, Any, Optional, Union, Tuple, List, Callable

import optuna
import gc

import warnings

warnings.filterwarnings('ignore')

import random


def seed_everything(seed):
	np.random.seed(seed)
	random.seed(seed)


seed_everything(seed=42)

class baseline:
	model_name = ['LGBM', 'CAT', 'XGB', 'Voting', 'TABNET', 'Ridge', 'LR']
	metrics = [
		'roc_auc',
		'accuracy',
		'f1',
		'precision',
		'recall',
		'rmse',
		'wmae',
		'rmsle',
		'mae',
		'r2',
		'mse',
		'mape',
		'custom',
	]
	problem_types = ['classification', 'regression']
	cv_types = ['SKF', 'KF', 'GKF', 'GSKF', 'RKF']

	def __init__(
		self,
		train_data,
		test_data=None,
		target_column=None,
		tf_vec=False,
		gpu=False,
		numpy_data=False,
		handle_date=False,
		ordinal_encoder=False,
		problem_type='classification',
		metric='roc_auc',
		seed=42,
		ohe_fe=False,
		label_encode=False,
		target_encode=False,
		n_splits=5,
		cat_features=None,
		num_classes=None,
		prob=False,
		stat_fe=None,
		logger: Optional[loguru.logger] = None,
		eval_metric_model=None,
		early_stop=False,
		test_prob=False,
		fold_type='SKF',
		weights=None,
		multi_column_tfidf=None,
		custom_metric=None,
	):
		self.train_data = train_data
		self.test_data = test_data
		self.target_column = target_column
		self.problem_type = problem_type
		self.metric = metric
		self.seed = seed
		self.n_splits = n_splits
		self.cat_features = cat_features if cat_features else []
		self.num_classes = num_classes
		self.prob = prob
		self.test_prob = test_prob
		self.early_stop = early_stop
		self.fold_type = fold_type
		self.weights = weights
		self.tf_vec = tf_vec
		self.stat_fe = stat_fe
		self.multi_column_tfidf = multi_column_tfidf
		self.gpu = gpu
		self.numpy_data = numpy_data
		self.handle_date = handle_date
		self.ohe_fe = ohe_fe
		self.label_encode = label_encode
		self.ordinal_encoder = ordinal_encoder
		self.target_encode = target_encode
		self.custom_metric = custom_metric
		self.eval_metric_model = eval_metric_model
		self.logger = logger or self._setup_default_logger()

		if self.metric == 'custom' and callable(self.custom_metric):
			self.metric_name = self.custom_metric.__name__
		else:
			self.metric_name = self.metric

		self._validate_input()

	def _validate_input(self):
		if not isinstance(self.train_data, pd.DataFrame):
			raise ValueError('Training data must be a pandas DataFrame.')
		if self.test_data is not None and not isinstance(self.test_data, pd.DataFrame):
			raise ValueError('Test data must be a pandas DataFrame.')
		if self.target_column not in self.train_data.columns:
			raise ValueError(f"Target column '{self.target_column}' not found in the training dataset.")
		if self.problem_type not in self.problem_types:
			raise ValueError("Invalid problem type. Choose either 'classification' or 'regression'.")
		if self.metric not in self.metrics and self.metric not in self.regression_metrics:
			raise ValueError('Invalid metric. Choose from available metrics.')
		if not isinstance(self.n_splits, int) or self.n_splits < 2:
			raise ValueError('n_splits must be an integer greater than 1.')
		if self.fold_type not in self.cv_types:
			raise ValueError(f'Invalid fold type. Choose from {self.cv_types}.')

	def weighted_mean_absolute_error(self, y_true, y_pred, weights):
		return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

	def rmsLe(self, y_true, y_pred):
		y_pred = np.maximum(y_pred, 1e-6)
		return np.sqrt(mean_squared_log_error(y_true, y_pred))

	def mape(self, y_true, y_pred):
		return mean_absolute_percentage_error(y_true, y_pred)

	def weighted_mean_absolute_error(self, y_true, y_pred, weights):
		return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

	def rmsLe(self, y_true, y_pred):
		y_pred = np.maximum(y_pred, 1e-6)
		return np.sqrt(mean_squared_log_error(y_true, y_pred))

	def mape(self, y_true, y_pred):
		return mean_absolute_percentage_error(y_true, y_pred)
