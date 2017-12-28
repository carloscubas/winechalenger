import pandas as pd
wine = pd.read_csv('winequality-red.csv',';')

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

X = wine.drop('quality' , 1).values # drop target variable
y = wine['quality'].values 

RANDOM_STATE = 42

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)

from sklearn import neighbors 
knn = neighbors.KNeighborsClassifier(n_neighbors = 5)
knn_model_1 = knn.fit(X_train, y_train)
print('k-NN accuracy for test set: %f' % knn_model_1.score(X_test, y_test))


std_clf = make_pipeline(StandardScaler(), GaussianNB(priors=None))
std_clf.fit(X_train, y_train)
pred_test_std = std_clf.predict(X_test)

print('\nPrevisão para o conjunto de dados')
print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test_std)))
