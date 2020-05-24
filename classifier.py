# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cluster, manifold, neighbors, linear_model, ensemble, neural_network, tree, svm
from sklearn import metrics, model_selection, preprocessing

from tqdm import tqdm

# %%
TRAIN_SAMPLES = 5000
TEST_SAMPLES = 1000
USE_TS_FEATURES = True

# %%
trainFeatures = pd.read_hdf('dataset/preprocessed-data.h5', key='all_validate')
trainLabels = pd.read_hdf('dataset/preprocessed-data.h5', key='flag_validate')
testFeatures = pd.read_hdf('dataset/preprocessed-data.h5', key='all_train')
testLabels = pd.read_hdf('dataset/preprocessed-data.h5', key='flag_train')

# %%
# if USE_TS_FEATURES:
#   trainFeatures = addTimeSeriesFeatures(trainFeatures)
#   testFeatures = addTimeSeriesFeatures(testFeatures)
# else:
#   trainFeatures = trainFeatures.drop('year', axis=1)
#   testFeatures = testFeatures.drop('year', axis=1)

trainSet = pd.merge(trainFeatures, trainLabels, how='inner', on='ID')
trainSet = trainSet.dropna().drop(['ID'], axis=1)
testSet = pd.merge(testFeatures, testLabels, how='inner', on='ID')
testSet = testSet.dropna().drop(['ID'], axis=1)

# %%
# prepare data
nindex, pindex = trainSet.groupby('flag').groups.values()
sampleSize = min(len(nindex), len(pindex), TRAIN_SAMPLES//2)
trainSetBanlanced = pd.concat([trainSet.loc[nindex].sample(sampleSize),
                               trainSet.loc[pindex].sample(sampleSize)], axis=0)
# self-labeling
X, y = trainSetBanlanced.drop('flag', axis=1), trainSetBanlanced.flag
Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(X, y, test_size=0.2)

# scaler = preprocessing.StandardScaler()
# Xtrain = scaler.fit_transform(trainSetBanlanced.drop('flag', axis=1))
# ytrain = trainSetBanlanced.flag

# # cross-dataset labeling
# testSet = testSet.sample(TEST_SAMPLES)
# Xtest, ytest = scaler.transform(testSet.drop('flag', axis=1)), testSet.flag


# %%
# try with different classifiers
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
models = [neighbors.KNeighborsClassifier(n_neighbors=5),
          linear_model.LogisticRegression(),
          svm.SVC(),
          tree.DecisionTreeClassifier(),
          neural_network.MLPClassifier(),
          ensemble.BaggingClassifier(),
          ensemble.RandomForestClassifier(),
          ensemble.AdaBoostClassifier(),
          ensemble.GradientBoostingClassifier()]
for ax, model in zip(axes.flatten(), models):
  clf = model.fit(Xtrain, ytrain)
  # clf = linear_model.LogisticRegression().fit(Xtrain, ytrain)
  metrics.plot_confusion_matrix(clf, Xtest, ytest, ax=ax, values_format='d')
  f1 = metrics.f1_score(y_true=ytest, y_pred=clf.predict(Xtest))
  ax.set(title=f'{type(clf).__name__}\nF1={f1:.2f}')

# %%
# visualization
feature2D = manifold.TSNE().fit_transform(Xtest, ytest)
plt.scatter(feature2D[:, 0], feature2D[:, 1], c=ytest)

# %%
# plot confusion matrix with ?NN
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for k, ax in zip(range(1, 10, 2), axes.flatten()):
  clf = neighbors.KNeighborsClassifier(n_neighbors=k).fit(Xtrain, ytrain)
  # clf = linear_model.LogisticRegression().fit(Xtrain, ytrain)
  metrics.plot_confusion_matrix(clf, Xtest, ytest, ax=ax, values_format='d')
  f1 = metrics.f1_score(y_true=ytest, y_pred=clf.predict(Xtest))
  ax.set(title=f'{k}NN, F1={f1:.2f}')


# %%
# set best K
knnscore = {}
for k in tqdm(range(1, 15, 2)):
  clf = neighbors.KNeighborsClassifier(n_neighbors=k)
  score = model_selection.cross_val_score(clf, X, y, cv=200, n_jobs=-1)
  knnscore[k] = score

plt.plot(list(knnscore.keys()), list(knnscore.values()))

# %%
plt.plot(list(knnscore.keys()), list((np.mean(s) for s in knnscore.values())))


# %%
