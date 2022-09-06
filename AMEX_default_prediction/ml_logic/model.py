#from colorama import Fore, Style
import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split


from AMEX_default_prediction.ml_logic.params import CAT_VARS
# Categorical Feature Support --> https://lightgbm.readthedocs.io/en/v3.3.2/Quick-Start.html
# LightGBM can use categorical features directly (without one-hot encoding). The experiment on Expo data shows about 8x speed-up compared with one-hot encoding.
# For the setting details, please refer to the categorical_feature parameter.


def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:

    ## TWEAK

    y_pred = pd.Series(y_pred, index=y_true.index)

    y_true = pd.DataFrame(y_true)
    y_pred = pd.DataFrame(y_pred)

    y_true = y_true.rename(columns={y_true.columns[0]:'target'})
    y_pred = y_pred.rename(columns={y_pred.columns[0]:'prediction'})
    ##

    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)

def build_param_dict(boosting_type='gbdt', # params can be found here https://lightgbm.readthedocs.io/en/v3.3.2/Parameters.html
                     max_depth=-1,
                     n_estimators=100,
                     num_leaves=31,
                     class_weight = None, # might want to try is_unbalance
                     learning_rate= 0.1,
                     min_data_in_leaf=20,
                     bagging_fraction=1.0,
                     feature_fraction=1.0,
                     objective='binary',
                     reg_alpha=0.,
                     reg_lambda=0.,
                     categorical_features=CAT_VARS,
                     random_state=None
                     ):

    return dict(
        boosting_type=boosting_type, # params can be found here https://lightgbm.readthedocs.io/en/v3.3.2/Parameters.html
        max_depth=max_depth,
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        class_weight = class_weight, # might want to try is_unbalance
        learning_rate= learning_rate,
        min_data_in_leaf=min_data_in_leaf,
        bagging_fraction=bagging_fraction,
        feature_fraction=feature_fraction,
        objective=objective,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        categorical_features=categorical_features,
        random_state=random_state
        )


def initialize_model(X: np.ndarray, param_dict=build_param_dict()):

    return lgb.LGBMClassifier(**param_dict) ## in the end the model seems to have some problems with categorical features.
## see also here https://www.kaggle.com/code/mlisovyi/beware-of-categorical-features-in-lgbm/notebook -- an ordinal encoder and not declaring the features categorical in the model helps

def train_model(model,
                X: np.ndarray,
                y: np.ndarray,
                eval_metric=make_scorer(amex_metric),
                early_stopping=30,
                num_splits=1,
                test_size = 0.3,
                init_model = None, #(str, pathlib.Path, Booster, LGBMModel or None, optional (default=None)) – Filename of LightGBM model, Booster instance or LGBMModel instance used for continue training.
                ):

    ## make sure the passed data is aggregated, otherwise it is going to split row/wise, not ustomer wise

    # check out init score and init model


    eval_sets = []

    for i in range(num_splits):

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)
        eval_sets.append((X_val, y_val))

    model.fit(X, y, early_stopping=early_stopping, eval_metric=eval_metric, eval_set=eval_sets, init_model=None)

    return model

def evaluate_model(model,X,y):

    pass
