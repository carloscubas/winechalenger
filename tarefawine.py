#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 21:17:57 2017

@author: cubas
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.datasets import load_wine

#wine = pd.read_csv('winequality-red.csv',';')

#X = wine.drop('quality' , 1).values # drop target variable
#y = wine['quality'].values 

X, y = load_wine(return_X_y=True)

X_normalized = preprocessing.normalize(X, norm='l2')
RANDOM_STATE = 42

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.30, random_state=RANDOM_STATE)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB(priors=None)
clf.fit(X_train, y_train)
pred_test_nb = clf.predict(X_test)

from sklearn.naive_bayes import MultinomialNB
multinomial_model = MultinomialNB()
multinomial_model.fit(X_train, y_train)
pred_test_multinomial = multinomial_model.predict(X_test)

from sklearn.ensemble import AdaBoostClassifier
adabost_model = AdaBoostClassifier()
adabost_model.fit(X_train, y_train)
pred_test_adabost = adabost_model.predict(X_test)

# Fit to data and predict using pipelined scaling, GNB and PCA.
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
std_clf = make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB())
std_clf.fit(X_train, y_train)
pred_test_std = std_clf.predict(X_test)

print('Previsão para o conjunto de dados Naive Bayes: {:.2%}'.format(metrics.accuracy_score(y_test, pred_test_nb, normalize = True)))
print('Previsão para o conjunto de dados Multinomial: {:.2%}'.format(metrics.accuracy_score(y_test, pred_test_multinomial, normalize = True)))
print('Previsão para o conjunto de dados Adabost: {:.2%}'.format(metrics.accuracy_score(y_test, pred_test_adabost, normalize = True)))
print('Previsão para o conjunto de dados PCA: {:.2%}'.format(metrics.accuracy_score(y_test, pred_test_std, normalize = True)))