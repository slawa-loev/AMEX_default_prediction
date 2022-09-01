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


#Deep learning
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping


# ## google cloud storage
# import logging
# import os
# import cloudstorage as gcs
# import webapp2
# from google.appengine.api import app_identity

## Define functions to pull from google cloud buckets

# def get_defaulter(self):
#   bucket_name = os.environ.get('defaulter_data_13364',
#                                app_identity.get_default_gcs_bucket_name())

#   self.response.headers['Content-Type'] = 'text/plain'
#   self.response.write('Demo GCS Application running from Version: '
#                       + os.environ['CURRENT_VERSION_ID'] + '\n')
#   self.response.write('Using bucket name: ' + bucket_name + '\n\n')


# def get_payer(self):
#   bucket_name = os.environ.get('payer_data_41940',
#                                app_identity.get_default_gcs_bucket_name())

#   self.response.headers['Content-Type'] = 'text/plain'
#   self.response.write('Demo GCS Application running from Version: '
#                       + os.environ['CURRENT_VERSION_ID'] + '\n')
#   self.response.write('Using bucket name: ' + bucket_name + '\n\n')


# def read_file(self, filename):
#   self.response.write('Reading the full file contents:\n')

#   gcs_file = gcs.open(filename)
#   contents = gcs_file.read()
#   gcs_file.close()
#   self.response.write(contents)

# # Pull data from google cloud
# def_df = read_file(get_defaulter())
# pay_df = read_file(get_payer())


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


def_df = pd.read_csv("/Users/sjoerddewit/Desktop/Programming/6 Le Wagon Data Science/final_project/defaulter_data_13364.csv", index_col=[0])
pay_df = pd.read_csv("/Users/sjoerddewit/Desktop/Programming/6 Le Wagon Data Science/final_project/payer_data_41940.csv", index_col=[0])

def_df['default'] = 1
pay_df['default'] = 0

df = pd.concat([def_df, pay_df])

y = df['default']

X = df.drop(columns=['default'])

cat_vars = ['B_30',
            'B_38',
            'D_114',
            'D_116',
            'D_117',
            'D_120',
            'D_126',
            'D_63',
            'D_64',
            'D_66',
            'D_68']

X_corr = X.corr()

X_corr = X_corr.unstack().reset_index() # Unstack correlation matrix
X_corr.columns = ['feature_1','feature_2', 'correlation_all'] # rename columns
X_corr.sort_values(by="correlation_all",ascending=False, inplace=True) # sort by correlation
X_corr = X_corr[X_corr['feature_1'] != X_corr['feature_2']] # Remove self correlation
X_corr = X_corr.drop_duplicates(subset='correlation_all')

red_features = list(X_corr[abs(X_corr['correlation_all'])>=.95]['feature_1']) ## abs so we also consider the negative corrs

X_red = X.drop(columns=red_features) ## dropping the highly correlated columns

## checking whether the high correlations are gone
X_red_corr = X_red.corr()
X_red_corr = X_red_corr.unstack().reset_index() # Unstack correlation matrix
X_red_corr.columns = ['feature_1','feature_2', 'correlation_all'] # rename columns
X_red_corr.sort_values(by="correlation_all",ascending=False, inplace=True) # sort by correlation
X_red_corr = X_red_corr[X_red_corr['feature_1'] != X_red_corr['feature_2']] # Remove self correlation
X_red_corr = X_red_corr.drop_duplicates(subset='correlation_all')


nan_threshold= 0.8 ## adjust the hardcoded values

def_nans = def_df.isna().sum()/len(def_df)
def_nans_80 = def_nans[def_nans >= 0.8].index
pay_nans = pay_df.isna().sum()/len(pay_df)
pay_nans_80 = pay_nans[pay_nans>=0.8].index

nans_80 = [feature for feature in pay_nans_80 if feature in def_nans_80]

## check whether features were already removed
red_features_nan = [feature for feature in nans_80 if feature not in red_features]

X_red = X_red.drop(columns=red_features_nan)

dropped_columns = red_features + red_features_nan

num_vars = [feature for feature in X_red.columns[2:] if feature not in cat_vars] ## exclude dates and IDs (first two columns)
str_vars = [feature for feature in X_red.columns[2:] if not pd.api.types.is_numeric_dtype(X_red[feature])] ## columns that are not numeric at all
red_cat_vars = [feature for feature in cat_vars if feature not in dropped_columns] ## remaining categorical variables that have no string values

def nan_imp(X): ## imputes nan values for alternative values signifying nans
    nan_list = [-1,-1.0, "-1.0", "-1"]
    return X.applymap(lambda x: np.nan if x in nan_list else x) ## perhaps subfunctions for arrays

# impute mean/most frequent value for other nans (specific to group?)
# robustscale all numerical values

num_scaler = RobustScaler()
num_imputer = SimpleImputer(strategy='mean')
#num_imputer = KNNImputer(n_neighbors=2) ## KNNIMputer is computationally demanding
## should come AFTER SCALING

num_pipe = make_pipeline(num_scaler, num_imputer)

str_trans = OrdinalEncoder() # is only needed if one wants to do knnimputer

nan_trans = FunctionTransformer(nan_imp)
cat_imputer = SimpleImputer(strategy="most_frequent") ## replace with KNNimputer on one neighbour, after transforming to numericals
#cat_imputer = KNNImputer(n_neighbors=1) # introducing it did not improve performance, but is computationally demanding
cat_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore') ## what happens to the old columns?
cat_pipe = make_pipeline(nan_trans, cat_imputer, cat_encoder)
str_pipe = make_pipeline(nan_trans, str_trans, cat_imputer, cat_encoder)


preprocessor = ColumnTransformer([
    ('num_pip', num_pipe, num_vars),
    ('cat_pip', cat_pipe, red_cat_vars),
    ('str_pip', str_pipe, str_vars)],
    remainder='drop' ## all columns not in num_vars and red_cat_vars are dropped.
)

X_pp = pd.DataFrame(preprocessor.fit_transform(X_red))




X_train, X_test, y_train, y_test = train_test_split(X_pp,
                                                    y,
                                                    test_size=0.30,
                                                    random_state=42)


######### Deep Learning model verison 1 ##############


model_sig = Sequential()
model_sig.add(layers.Dense(64, activation='relu'))
model_sig.add(layers.Dense(32, activation='relu'))
model_sig.add(layers.Dense(32, activation='relu'))
model_sig.add(layers.Dense(16, activation='relu'))
model_sig.add(layers.Dense(1, activation='sigmoid'))


# Compilation
model_sig.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


# Early Stopping
es = EarlyStopping(patience=10, restore_best_weights=True)


# Fit
model_sig.fit(X_train,
              y_train,
              batch_size=100,
              epochs=100,
              callbacks=[es],
              validation_split=0.3,
              verbose=1)


# Evaluation


rand_ints = list(range(0,100))
results = []
y_true = []

for num in rand_ints[:10]:
    x_sample = X_test.sample(random_state=num)
    y_sample = y_test.sample(random_state=num)

    results.append(model_sig.evaluate(x_sample, y_sample)[1])
    print(y_sample[y_sample.index[0]])
    y_true.append(y_sample[y_sample.index[0]])
