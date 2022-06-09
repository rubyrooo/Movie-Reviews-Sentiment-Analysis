"""xgb.py: Copyright 2020, Sentiment Analysis on Movie Reviews using LSTM"""
__authors__ = "Xueru Ye, Ruoran Liu, Keith Herbert"
__copyright__ = "Copyright 2020, Sentiment Analysis on Movie Reviews"
__license__ = "GPL"
__version__ = "1.0.0"
__maintained_by__ = "Xueru Ye, Ruoran Liu, Keith Herbert"
__email__ = "xye85@uwo.ca@uwo.ca, rliu454@uwo.ca, kherbe@uwo.ca"
__status__ = "Production"

import numpy as np
import pandas as pd
from memory_profiler import profile
from time import perf_counter
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
import sys

df_train = pd.read_csv('data.tsv', delimiter='\t')
df_train = df_train.sample(n = 2000)
df_train.fillna('null',inplace=True)

x_train, x_test, y_train, y_test = train_test_split(df_train['Phrase'], df_train['Sentiment'], test_size=0.25)

vectorizer = CountVectorizer(max_features=5000)
tf_idf_transformer = TfidfTransformer()
tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(x_train))
x_train_weight = tf_idf.toarray()  # TF-IDF matrix for training set
tf_idf = tf_idf_transformer.transform(vectorizer.transform(x_test))
x_test_weight = tf_idf.toarray()  # TF-IDF matrix for testing set

random_seed = 8
kfold_scoring = 'accuracy'
kfold_n_splits = 5
kfold_result_output = "%s KFold Validation: Mean %f (STD %f)"
model_name = 'XGBoost Model'

xlf = xgb.XGBClassifier(max_depth=6,
                learning_rate=0.51,
                n_estimators=8,
                silent=True,
                objective='multi:softmax',
                num_class = 5,
                nthread=-1,
                gamma=0,
                min_child_weight=1,
                max_delta_step=0,
                subsample=0.85,
                colsample_bytree=0.7,
                colsample_bylevel=1,
                reg_alpha=0,
                reg_lambda=1,
                scale_pos_weight=1,
                seed=1440,
                missing=None)

sys.setrecursionlimit(10000)
#@profile
def fit_predict(xlf_temp):
    t1_start = perf_counter()
    xlf.fit(x_train_weight, y_train, eval_metric='merror', verbose=True, eval_set=[(x_test_weight,  y_test)], early_stopping_rounds=100)
    t1_stop = perf_counter()
    print("Elapsed training time: ", t1_stop - t1_start)
    
    t1_start = perf_counter()
    y_pred = xlf.predict(x_test_weight, ntree_limit=xlf.best_ntree_limit)
    t1_stop = perf_counter()
    print("Elapsed Predicting time: ", t1_stop - t1_start)
    
    return y_pred

y_pred = fit_predict(xlf)

print(cross_val_score(xlf, x_train_weight, y_train, cv=5))

label_all = ['0', '1','2','3','4']
confusion_mat = metrics.confusion_matrix(y_test, y_pred)
df = pd.DataFrame(confusion_mat, columns=label_all)
df.index = label_all
print('accuracy', metrics.accuracy_score(y_test, y_pred))
print('confusion_matrix:\n', df)
print('classification report:\n', metrics.classification_report(y_test, y_pred))