# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 18:54:23 2016

@author: shen
"""

import time
start_time = time.time()

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import pipeline, grid_search
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
from nltk.stem.snowball import SnowballStemmer #0.003 improvement but takes twice as long as PorterStemmer
stemmer = SnowballStemmer('english')
import re
import random
random.seed(31337)
#from nltk.corpus import stopwords
import random
import re, math
from collections import Counter

df_train = pd.read_csv('./Kag_HDS/input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('./Kag_HDS/input/test.csv', encoding="ISO-8859-1")
df_pro_desc = pd.read_csv('./Kag_HDS/input/product_descriptions.csv')
df_attr = pd.read_csv('./Kag_HDS/input/attributes.csv')
df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
df_bullet1 = df_attr[df_attr.name == "Bullet01"][["product_uid", "value"]].rename(columns={"value": "bullet1"})
df_bullet2 = df_attr[df_attr.name == "Bullet02"][["product_uid", "value"]].rename(columns={"value": "bullet2"})
df_bullet3 = df_attr[df_attr.name == "Bullet03"][["product_uid", "value"]].rename(columns={"value": "bullet3"})
df_bullet4 = df_attr[df_attr.name == "Bullet04"][["product_uid", "value"]].rename(columns={"value": "bullet4"})
num_train = df_train.shape[0]
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
df_all = pd.merge(df_all, df_bullet1, how='left', on='product_uid')
df_all = pd.merge(df_all, df_bullet2, how='left', on='product_uid')
df_all = pd.merge(df_all, df_bullet3, how='left', on='product_uid')
df_all = pd.merge(df_all, df_bullet4, how='left', on='product_uid')

strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}


def str_stem(s):
    if isinstance(s, str):
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
        s = s.lower()
        s = s.replace("  "," ")
        s = s.replace(", ","") #could be number / segment later
        s = s.replace("$"," ")
        s = s.replace("?"," ")
        s = s.replace("-"," ")
        s = s.replace("//","/")
        s = s.replace("..",".")
        s = s.replace(" / "," ")
        s = s.replace(" \\ "," ")
        s = s.replace("."," . ")
        s = s.replace("′′"," inches ")
        s = s.replace("′"," feet ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x "," xbi ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*"," xbi ")
        s = s.replace(" by "," xbi ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|'')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gals|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(millimeters|milimeters|mm)\.?", r"\1mm. ", s)
        s = s.replace("°"," degrees ")
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
        s = s.replace(" v "," volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        s = s.replace("  "," ")
        s = s.replace(" . "," ")
        s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])

        s = s.lower()
        s = s.replace("toliet","toilet")
        s = s.replace("airconditioner","air conditioner")
        s = s.replace("condtion","condition")
        s = s.replace("vinal","vinyl")
        s = s.replace("vynal","vinyl")
        s = s.replace("skill","skil")
        s = s.replace("snowbl","snow bl")
        s = s.replace("plexigla","plexi gla")
        s = s.replace("rustoleum","rust-oleum")
        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool ga")
        s = s.replace("whirlpoolstainless","whirlpool stainless")
        return s
    else:
        return "null"

def seg_words(str1, str2):
    str2 = str2.lower()
    str2 = re.sub("[^a-z0-9./]"," ", str2)
    str2 = [z for z in set(str2.split()) if len(z)>2]
    words = str1.lower().split(" ")
    s = []
    for word in words:
        if len(word)>3:
            s1 = []
            s1 += segmentit(word,str2,True)
            if len(s)>1:
                s += [z for z in s1 if z not in ['er','ing','s','less'] and len(z)>1]
            else:
                s.append(word)
        else:
            s.append(word)
    return (" ".join(s))

def segmentit(s, txt_arr, t):
    st = s
    r = []
    for j in range(len(s)):
        for word in txt_arr:
            if word == s[:-j]:
                r.append(s[:-j])
                #print(s[:-j],s[len(s)-j:])
                s=s[len(s)-j:]
                r += segmentit(s, txt_arr, False)
    if t:
        i = len(("").join(r))
        if not i==len(st):
            r.append(st[i:])
    return r

def str_common_word(str1, str2):
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word)>=0:
            cnt+=1
    return cnt

def str_whole_word(str1, str2, i_):
    cnt = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt

#cachedStopWords = stopwords.words("english")
WORD = re.compile(r'\w+')

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)

def calculate_similarity(str1,str2):
    vector1 = text_to_vector(str1)
    vector2 = text_to_vector(str2)
    return get_cosine(vector1, vector2)

'''def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in cachedStopWords])
'''
class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        d_col_drops=['id','relevance','search_term','product_title',
            'product_description','product_info','attr','brand','bullets',
            'bullet1','bullet2','bullet3','bullet4', 'len_of_bullet1',
            'len_of_bullet2', 'len_of_bullet3', 'len_of_bullet4']
        hd_searches = hd_searches.drop(d_col_drops,axis=1).values
        return hd_searches

class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key].apply(str)

def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)

print("--- Start Stemming: %s minutes ---" % round(((time.time() - start_time)/60),2))
df_all['search_term'] = df_all['search_term'].map(lambda x:str_stem(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stem(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stem(x))
df_all['brand'] = df_all['brand'].map(lambda x:str_stem(x))
df_all['bullet1'] = df_all['bullet1'].map(lambda x:str_stem(x))
df_all['bullet2'] = df_all['bullet2'].map(lambda x:str_stem(x))
df_all['bullet3'] = df_all['bullet3'].map(lambda x:str_stem(x))
df_all['bullet4'] = df_all['bullet4'].map(lambda x:str_stem(x))

print("--- Finish Stemming: %s minutes ---" % round(((time.time() - start_time)/60),2))

df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title'] +"\t"+df_all['product_description']
print("--- Prod Info: %s minutes ---" % round(((time.time() - start_time)/60),2))
df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_title'] = df_all['product_title'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_description'] = df_all['product_description'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_brand'] = df_all['brand'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_bullet1'] = df_all['bullet1'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_bullet2'] = df_all['bullet2'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_bullet3'] = df_all['bullet3'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_bullet4'] = df_all['bullet4'].map(lambda x:len(x.split())).astype(np.int64)
print("--- Len of: %s minutes ---" % round(((time.time() - start_time)/60),2))
df_all['search_term'] = df_all['product_info'].map(lambda x:seg_words(x.split('\t')[0],x.split('\t')[1]))
#print("--- Search Term Segment: %s minutes ---" % round(((time.time() - start_time)/60),2))
df_all['query_in_title'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[1],0))
df_all['query_in_description'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[2],0))
print("--- Query In: %s minutes ---" % round(((time.time() - start_time)/60),2))
df_all['query_last_word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1],x.split('\t')[1]))
df_all['query_last_word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1],x.split('\t')[2]))
print("--- Query Last Word In: %s minutes ---" % round(((time.time() - start_time)/60),2))
df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
df_all['ratio_title'] = df_all['word_in_title']/df_all['len_of_query']
df_all['ratio_description'] = df_all['word_in_description']/df_all['len_of_query']
df_all['attr'] = df_all['search_term']+"\t"+df_all['brand']
df_all['word_in_brand'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['ratio_brand'] = df_all['word_in_brand']/df_all['len_of_brand']
df_all['bullets'] = df_all['search_term']+"\t"+df_all['bullet1']+"\t"+df_all['bullet2']+"\t"+df_all['bullet3']+"\t"+df_all['bullet4']
df_all['word_in_bullet1'] = df_all['bullets'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['ratio_bullet1'] = df_all['word_in_bullet1']/df_all['len_of_bullet1']
df_all['word_in_bullet2'] = df_all['bullets'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
df_all['ratio_bullet2'] = df_all['word_in_bullet2']/df_all['len_of_bullet2']
df_all['word_in_bullet3'] = df_all['bullets'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[3]))
df_all['ratio_bullet3'] = df_all['word_in_bullet3']/df_all['len_of_bullet3']
df_all['word_in_bullet4'] = df_all['bullets'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[4]))
df_all['ratio_bullet4'] = df_all['word_in_bullet4']/df_all['len_of_bullet4']
df_brand = pd.unique(df_all.brand.ravel())
d={}
i = 1000
for s in df_brand:
    d[s]=i
    i+=3
df_all['brand_feature'] = df_all['brand'].map(lambda x:d[x])
df_all['search_term_feature'] = df_all['search_term'].map(lambda x:len(x))

df_all['similarity_in_description']=df_all['product_info'].map(lambda x:calculate_similarity(x.split('\t')[0],x.split('\t')[2]))
df_all['similarity_in_title']=df_all['product_info'].map(lambda x:calculate_similarity(x.split('\t')[0],x.split('\t')[1]))

df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id']
y_train = df_train['relevance'].values
X_train =df_train[:]
X_test = df_test[:]
print("--- Features Set: %s minutes ---" % round(((time.time() - start_time)/60),2))

rfr = RandomForestRegressor(n_estimators = 500, random_state = 31337, verbose = 1)
tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
tsvd = TruncatedSVD(n_components=15, random_state = 2016)
clf = pipeline.Pipeline([
        ('union', FeatureUnion(
                    transformer_list = [
                        ('cst',  cust_regression_vals()),
                        ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='search_term')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                        ('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='product_title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                        ('txt4', pipeline.Pipeline([('s4', cust_txt_col(key='brand')), ('tfidf4', tfidf), ('tsvd4', tsvd)]))
                        ]
                )),
        ('rfr', rfr)])
param_grid = {
                'rfr__n_estimators' : [500],
                'rfr__max_depth': [20],
                'rfr__max_features' : [15]
             }
model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid,
    cv = 10, verbose = 250, scoring=RMSE)
model.fit(X_train, y_train)

print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)

y_pred = model.predict(X_test)
print(len(y_pred))
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('./Kag_HDS/output/attr_submit.csv',index=False)
print("--- Training & Testing: %s minutes ---" % ((time.time() - start_time)/60))