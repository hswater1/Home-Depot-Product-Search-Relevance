# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:18:44 2016

@author: shen
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:39:57 2016

@author: Gordon
"""
import time
start_time = time.time()

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
#from sklearn import pipeline, model_selection
from sklearn import pipeline, grid_search
#from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
#from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
#from nltk.metrics import edit_distance
#from nltk.stem.porter import PorterStemmer
#stemmer = PorterStemmer()
from nltk.stem.snowball import SnowballStemmer #0.003 improvement but takes twice as long as PorterStemmer
stemmer = SnowballStemmer('english')

import random
random.seed(22)

def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)

class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        d_col_drops=['title2', 'id','relevance','search_term','product_title','product_description','product_info','attr','brand',]
        hd_searches = hd_searches.drop(d_col_drops,axis=1).values
        return hd_searches

class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key].apply(str)

df_train = pd.read_csv('C:/Homedepot/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('C:/Homedepot/test.csv', encoding="ISO-8859-1")
#df_pro_desc = pd.read_csv('C:/Homedepot/product_descriptions.csv')
#df_attr = pd.read_csv('C:/Homedepot/attributes.csv')
#df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
num_train = df_train.shape[0]
print("--- Files Loaded: %s minutes ---" % round(((time.time() - start_time)/60),2))


#df_all = pd.read_csv('C:/Homedepot/sandbox/df_all_owl.csv', encoding="ISO-8859-1", index_col=0)
#df_substr = pd.read_csv('C:/Homedepot/sandbox/test_substr.csv', encoding="ISO-8859-1", index_col=0)
#df_substr = df_substr[['id', 'relevance']]
#df_substr.columns = ['id','substr_title']
#df_substr['relevance']= np.where(df_substr['relevance'] == 1,1,0)
#df_all = pd.merge(df_all, df_substr, how='left', on='id')
#print(df_all[:10])
df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id']
y_train = df_train['relevance'].values
X_train =df_train[:]
X_test = df_test[:]
print("--- Features Set: %s minutes ---" % round(((time.time() - start_time)/60),2))
rfr = RandomForestClassifier(n_estimators = 500, n_jobs = -1, random_state = 2016, verbose = 1)
tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
tsvd = TruncatedSVD(n_components=15, random_state = 2016)
clf = pipeline.Pipeline([
        ('union', FeatureUnion(
                    transformer_list = [
                        ('cst',  cust_regression_vals()),  
                        ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='search_term')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                        ('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='product_title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                        ('txt3', pipeline.Pipeline([('s3', cust_txt_col(key='product_description')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
                        ('txt4', pipeline.Pipeline([('s4', cust_txt_col(key='brand')), ('tfidf4', tfidf), ('tsvd4', tsvd)]))
                        ],
                    transformer_weights = {
                        'cst': 1.0,
                        'txt1': 0.5,
                        'txt2': 0.25,
                        'txt3': 0.0,
                        'txt4': 0.5
                        },
                n_jobs = 1
                )), 
        ('rfr', rfr)])
param_grid = {'rfr__max_features': [15], 'rfr__max_depth': [20]}
model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = -1, cv = 2, verbose = 20, scoring=RMSE)
model.fit(X_train, y_train)

print("Best parameters found by grid search:")
print(model.best_estimator_)
print("Best CV score:")
print(model.best_score_)
print(model.best_score_ + 0.47003199274)
y_pred = model.predict(X_test)
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('./Kag_HDS/output/submission_0226_n_18_d_20.csv',index=False)
print("--- Training & Testing: %s minutes ---" % round(((time.time() - start_time)/60),2))