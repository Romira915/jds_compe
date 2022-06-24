#!/usr/bin/env python
# coding: utf-8

# In[34]:


import glob
import os
import random

import ngram
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, make_scorer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB

# In[35]:


def tokenize(word):
    word_list = []
    for n in range(2, 4):
        index = ngram.NGram(N=n)
        for w in index.ngrams(index.pad(word)):
            if w.find("$") != -1:
                continue
            word_list.append(w)
    return word_list


# In[36]:


train_df = pd.read_csv("./train.csv")
test_df = pd.read_csv("./test.csv")
compe_df = pd.read_csv("./compe.csv")


# In[40]:


print(compe_df)


# In[13]:


train_text = train_df["text"].values.astype('U')
test_text = test_df["text"].values.astype('U')
y = train_df["label"].values.astype("int8")
test_y = test_df["label"].values.astype("int8")
compe_text = compe_df["text"].values.astype('U')


# In[14]:


vectorizer = TfidfVectorizer(analyzer=tokenize)
vectorizer.fit(train_text)
X = vectorizer.transform(train_text)
test_vec = vectorizer.transform(test_text)
compe_vec = vectorizer.transform(compe_text)


# In[16]:


valid_scores = []

train_check = np.zeros(len(y), dtype=np.float64)
models = []

kf = KFold(n_splits=3)
for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
    x_train, y_train = X[tr_idx], y[tr_idx]
    x_valid, y_valid = X[va_idx], y[va_idx]
    m = GaussianNB()
    m.fit(x_train.toarray(), y_train)
    p = m.predict(x_valid)
    train_check[va_idx] += p
    models.append(m)

print(accuracy_score(y, np.where(train_check < 0.5, 0, 1)))

# In[37]:


result = np.zeros(len(test_y), dtype=np.float64)
result = np.array([model.predict(test_vec) for model in models]).mean(axis=0)
result = np.where(result < 0.5, 0, 1)


# In[48]:


print(accuracy_score(test_y, result))  # リーダーボードのスコア この値を毎回，discord上に載せてください．


# In[41]:


print(len(result))


# In[45]:


c_result = np.zeros(len(compe_text), dtype=np.float64)
c_result = np.array([model.predict(compe_vec)
                    for model in models]).mean(axis=0)
c_result = np.where(c_result < 0.5, 0, 1)
print(len(c_result))


# In[46]:


csv_data = pd.DataFrame(data=c_result, columns=["label"])
csv_data.reset_index(inplace=True)
csv_data = csv_data.rename(columns={'index': 'ID'})
# submission.csvを最大2つまで馬場宛に提出してください．良い方の結果を最終スコアとします．
csv_data.to_csv("./submission.csv", index=False)
