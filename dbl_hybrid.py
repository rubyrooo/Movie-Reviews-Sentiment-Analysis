"""dbl_hybrid.py: Copyright 2020, Sentiment Analysis on Movie Reviews using LSTM"""
__authors__ = "Xueru Ye, Ruoran Liu, Keith Herbert"
__copyright__ = "Copyright 2020, Sentiment Analysis on Movie Reviews"
__license__ = "GPL"
__version__ = "1.0.0"
__maintained_by__ = "Xueru Ye, Ruoran Liu, Keith Herbert"
__email__ = "xye85@uwo.ca@uwo.ca, rliu454@uwo.ca, kherbe@uwo.ca"
__status__ = "Production"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import model_selection

from memory_profiler import profile
from time import perf_counter

is_grid_search = True

label_field = "Sentiment"
phrase_field = "Phrase"

df = pd.read_csv('data.tsv', delimiter='\t')
df.drop(['PhraseId', 'SentenceId'], axis=1, inplace=True)
batch_1 = df[:2000]
#print(df.head(5))
print(batch_1[label_field].value_counts())

# DistilBERT Model
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

# Initialize Tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

tokenized = batch_1[phrase_field].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

np.array(padded).shape
attention_mask = np.where(padded != 0, 1, 0)
attention_mask.shape

input_ids = torch.tensor(padded).to(torch.int64)
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

features = last_hidden_states[0][:,0,:].numpy()
labels = batch_1[label_field]

train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

model_function = None
if is_grid_search == True:
    # Grid Search
    parameters = {'C': np.linspace(0.1, 10, 200)}
    model_function = GridSearchCV(LogisticRegression(), parameters)
else:
    # Linear Regression
    model_function = LogisticRegression()

# Validation Parameters
random_seed = 8
kfold_scoring = 'accuracy'
kfold_n_splits = 5
kfold_result_output = "%s KFold Validation: Mean %f (STD %f)"
model_name = 'DistilBERT Hybrid'

@profile
def fit_predict():
    t1_start = perf_counter()  
    model_function.fit(train_features, train_labels)
    t1_stop = perf_counter() 
    print("Elapsed Training Time:", t1_stop-t1_start) 

    t1_start = perf_counter()  
    model_predict = model_function.predict(test_features)
    t1_stop = perf_counter() 
    print("Elapsed Predicting Time:", t1_stop-t1_start) 

    return model_predict

model_predict = fit_predict()

# KFold Validation
kfold = model_selection.KFold(n_splits=5, random_state=random_seed)
kfold_result = model_selection.cross_val_score(model_function, features, labels, cv=kfold, scoring=kfold_scoring)
kfold_result_summary = kfold_result_output % (model_name, kfold_result.mean(), kfold_result.std())
print(kfold_result_summary)

# Hold-out Validation
model_predict = model_function.predict(test_features)
print ('\nHoldout Validation - Accuracy Score')
print(accuracy_score(test_labels, model_predict))
print ('\nHoldout Validation - Confusion Matrix')
print(confusion_matrix(test_labels, model_predict))
print ('\nHoldout Validation - Classification Report')
print(classification_report(test_labels, model_predict))

if is_grid_search == True:
    print('Grid Search - Best Training Parameters: ', model_function.best_params_)
    print('Grid Search - Best Training Score: ', model_function.best_score_)

# Reference - https://towardsdatascience.com/distilling-bert-how-to-achieve-bert-performance-using-logistic-regression-69a7fc14249d
# Reference - https://huggingface.co/transformers/examples.html#glue
# Reference - https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/