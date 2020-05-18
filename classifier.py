# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model, svm, ensemble, neural_network
from sklearn import preprocessing, model_selection, metrics

# %%
sns.set(font='SimHei')

# %%
# TODO: classifier class
class classifier():
  pass

# %%
corpdf = pd.read_hdf('dataset/preprocessed-data.h5', key='corp_validate')
financedf = pd.read_hdf('dataset/preprocessed-data.h5', key='finance_validate')


# %%
# prepare the corp. basic info
corpdf = corpdf.dropna()
X, y = corpdf.drop(['ID', 'flag'], axis=1), corpdf.flag
X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# %%
# prepare the money info
financedf = financedf.dropna()
X, y = financedf.drop(['ID', 'year', 'flag'], axis=1), financedf.flag
X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# %%
clf = linear_model.LogisticRegression()
# clf = neural_network.MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu')
# clf = svm.SVC()
# clf = ensemble.GradientBoostingClassifier()

clf.fit(X_train, y_train)
clf.score(X_train, y_train)

# %%
metrics.plot_roc_curve(clf, X_test, y_test)

# %%
metrics.confusion_matrix(y_test, clf.predict(X_test))

# %%
