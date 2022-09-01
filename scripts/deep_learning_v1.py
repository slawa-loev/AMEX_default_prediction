import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict, learning_curve,\
train_test_split, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import plot_confusion_matrix, classification_report, precision_recall_curve
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor, SGDClassifier, Ridge, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.compose import make_column_selector

## pipeline stuff

from sklearn.pipeline import Pipeline, make_pipeline, make_union
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn import set_config; set_config(display='diagram')

from sklearn.base import TransformerMixin, BaseEstimator

## google cloud storage
import logging
import os
import cloudstorage as gcs
import webapp2
from google.appengine.api import app_identity


class CustomOHE(TransformerMixin, BaseEstimator):

    def __init__(self):
        pass

    def fit(self, X):
        X_dummified = X.astype(str)
        X_dummified = X_dummified.applymap(lambda x: x if x not in ["nan", "NaN", "NAN", "Nan", "-1", "-1.0"] else float("nan"))
        X_dummified = pd.get_dummies(X_dummified)
        self.columns = X_dummified.columns
        return self

    def transform(self, X):
        X_dummified = X.astype(str)
        X_dummified = X_dummified.applymap(lambda x: x if x not in ["nan", "NaN", "NAN", "Nan", "-1", "-1.0"] else float("nan"))
        X_dummified = pd.get_dummies(X_dummified)
        # Only keep columns that are computed in the fit() method
        # Drop new dummy columns if new category appears in the test set that were never seen in train set
        X_dummified_reindexed = X_dummified.reindex(columns=self.columns, fill_value=0)
        return X_dummified_reindexed


# Define functions to pull from google cloud buckets

def get_defaulter(self):
  bucket_name = os.environ.get('defaulter_data_13364',
                               app_identity.get_default_gcs_bucket_name())

  self.response.headers['Content-Type'] = 'text/plain'
  self.response.write('Demo GCS Application running from Version: '
                      + os.environ['CURRENT_VERSION_ID'] + '\n')
  self.response.write('Using bucket name: ' + bucket_name + '\n\n')


def get_payer(self):
  bucket_name = os.environ.get('payer_data_41940',
                               app_identity.get_default_gcs_bucket_name())

  self.response.headers['Content-Type'] = 'text/plain'
  self.response.write('Demo GCS Application running from Version: '
                      + os.environ['CURRENT_VERSION_ID'] + '\n')
  self.response.write('Using bucket name: ' + bucket_name + '\n\n')


def read_file(self, filename):
  self.response.write('Reading the full file contents:\n')

  gcs_file = gcs.open(filename)
  contents = gcs_file.read()
  gcs_file.close()
  self.response.write(contents)

# Pull data from google cloud

# def_df = pd.read_csv("/home/slawa/code/code-rep0/projects/data/defaulter_data_13364.csv", index_col=[0])
# pay_df = pd.read_csv("/home/slawa/code/code-rep0/projects/data/payer_data_41940.csv", index_col=[0])

def_df = read_file(get_defaulter())
pay_df = read_file(get_payer())

def_df['default'] = 1
pay_df['default'] = 0

df = pd.concat([def_df, pay_df])
