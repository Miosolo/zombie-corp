# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import cluster, manifold, neighbors, linear_model, ensemble, neural_network, tree, svm
from sklearn import metrics, model_selection, preprocessing, feature_selection
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Decision Tree Viz
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.externals.six import StringIO
import pydotplus
import graphviz
from IPython.display import SVG

from sklearn.externals import joblib

# %%
def univarTest(df, flag, method='chi2'):
  df = pd.merge(df, flag, on='ID').drop('ID', axis=1).dropna()
  df.iloc[:, :-1] = preprocessing.minmax_scale(df.iloc[:, :-1])
  methods = {
    'chi2': feature_selection.chi2,
    'f': feature_selection.f_classif,
    'mutal': feature_selection.mutual_info_classif
  }
  if method == 'mutal':
    score = methods[method](df.iloc[:, :-1], df.flag)
    sig = pd.DataFrame({'feature': df.columns[:-1], method: score, 'score': score})
  else:
    score, pv = methods[method](df.iloc[:, :-1], df.flag)
    sig = pd.DataFrame({'feature': df.columns[:-1], method: score, 'p': pv})
    sig['score'] = -np.log(sig.p)
  return sig.sort_values('score')

def plotFeatureSig(sig):
  sig.plot.barh(x='feature', y='score', grid=True, figsize=(10, 20))

def decisionTreeViz(model):
  dot_data = StringIO()
  export_graphviz(model, feature_names=features, class_names=['正常企业', '僵尸企业'],
                  out_file=dot_data, filled=True, rounded=True, special_characters=True)
  graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
  SVG(graph.create_svg())

def tryAllClassifers(Xtrain, Xtest, ytrain, ytest):
  # try with different classifiers
  _, axes = plt.subplots(3, 3, figsize=(12, 12))
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

def tryClassifiersCV(X, y, models=None, cv=10):
  # cross validation
  if models is None:
    models = [linear_model.LogisticRegression(),
              tree.DecisionTreeClassifier(),
              ensemble.GradientBoostingClassifier()]
  scores = {}
  for model in models:
    score = model_selection.cross_val_score(model, X, y, cv=10, n_jobs=-1)
    modelName = type(model).__name__
    scores[modelName] = score
    plt.plot(score, label=modelName)
  plt.legend()


# %%
# # validate -> train
# trainFeatures = pd.read_hdf('dataset/preprocessed-data.h5', key='all_validate')
# trainFeatures = pd.concat([trainFeatures,
#                            pd.read_hdf('dataset/preprocessed-data.h5', key='all_validate2')])
# trainLabels = pd.read_hdf('dataset/preprocessed-data.h5', key='flag_validate')
# trainLabels = pd.concat([trainLabels, 
#                          pd.read_hdf('dataset/preprocessed-data.h5', key='flag_validate2')])
# testFeatures = pd.read_hdf('dataset/preprocessed-data.h5', key='all_train')
# testLabels = pd.read_hdf('dataset/preprocessed-data.h5', key='flag_train')

# train -> validate
trainFeatures = pd.read_hdf('dataset/preprocessed-data.h5', key='all_train')
trainLabels = pd.read_csv('dataset/infer-train-flag.csv')
testFeatures = pd.read_hdf('dataset/preprocessed-data.h5', key='all_validate')
testLabels = pd.read_hdf('dataset/preprocessed-data.h5', key='flag_validate')

inferenceFeatures = pd.read_hdf('dataset/preprocessed-data.h5', key='all_test')

# inferenceLabels = pd.read_hdf('dataset/preprocessed-data.h5', key='all_test')

# TRAIN_SAMPLES = 4000
# TEST_SAMPLES = 1000

# %%
# feature selection
features = set()

# # drop features with ultra low variance
# scaledFeatures = preprocessing.minmax_scale(trainFeatures, axis=0)
# varTest = pd.DataFrame({'feature': trainFeatures.columns, 
#                         'score': np.var(scaledFeatures, axis=0)}).sort_values('score')
# # selected threshold: 0.01
# # features -= set(var[var<0.01].index)

# add features with high correlation with the flag
chi2Test = univarTest(trainFeatures, trainLabels, 'chi2')
# threshold: 0.01 (p)
features |= set(chi2Test.feature[chi2Test.p < 0.01])

fTest = univarTest(trainFeatures, trainLabels, 'f')
# threshold: 0.001 (p)
features |= set(fTest.feature[fTest.p < 0.001])

mutalTest = univarTest(trainFeatures, trainLabels, 'mutal')
# threshold: 0.2 (mutal-information)
features |= set(mutalTest.feature[mutalTest.score > 0.2])

# %%
trainFeatures = trainFeatures[list(features) + ['ID']]
testFeatures = testFeatures[list(features) + ['ID']]
inferenceFeatures = inferenceFeatures[list(features) + ['ID']]

trainSet = pd.merge(trainFeatures, trainLabels, how='inner', on='ID')
trainSet = trainSet.dropna().set_index('ID')
testSet = pd.merge(testFeatures, testLabels, how='inner', on='ID')
testSet = testSet.set_index('ID')

features = trainSet.columns[:-1]

# %%
# prepare data

# trainSet = trainSet.sample(TRAIN_SAMPLES)

X, y = trainSet.drop('flag', axis=1), trainSet.flag

# X, y = RandomUnderSampler().fit_resample(X, y)
X, y = SMOTE().fit_resample(X, y)
# self-labeling
# Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(X, y, test_size=0.2)

# # %%
# # compare up and down sampling
# # up-sampling
# XtrainUp, ytrainUp = SMOTE().fit_resample(X, y)
# # under-sampling
# XtrainDown, ytrainDown = RandomUnderSampler().fit_resample(X, y)

# model = linear_model.LogisticRegression()
# plt.plot(model_selection.cross_val_score(model, XtrainUp, ytrainUp, cv=100, n_jobs=-1), label='upsampling')
# plt.plot(model_selection.cross_val_score(model, XtrainDown, ytrainDown, cv=100, n_jobs=-1), label='downsampling')
# plt.legend()

# %%
# rough test for GBDT
pipe = Pipeline([('scaler', preprocessing.StandardScaler()),
                 ('GBDT', ensemble.GradientBoostingClassifier(n_estimators=110, learning_rate=0.2, 
                                                              max_depth=4, min_samples_leaf=125, 
                                                              min_samples_split=196, subsample=0.57))])
# cross-validation
selfScores = model_selection.cross_val_score(pipe, X, y, cv=10, n_jobs=-1)
# on test set
pipe.fit(X, y)

testSet = testSet.dropna()
Xtest, ytest = testSet.drop('flag', axis=1), testSet.flag
metrics.plot_confusion_matrix(pipe, Xtest, ytest)
metrics.plot_roc_curve(pipe, Xtest, ytest)

# %%
joblib.dump(pipe, 'models/GBDT-pipe-from-train.pkl')
inference = pipe.predict(inferenceFeatures.drop('ID', axis=1))
inferenceFlag = pd.DataFrame({'ID': inferenceFeatures.ID, 'flag': inference})
inferenceFlag.flag = inferenceFlag.flag.astype('int')
inferenceFlag.to_csv('results/infer-test-flag-from-train.csv', index=False)

# %%
# GBDT param searching
paramScores = {}
# gbmBaseline = ensemble.GradientBoostingClassifier()
treeBaseline = tree.DecisionTreeClassifier()
prediction = model_selection.cross_val_predict(treeBaseline, X, y, cv=10, n_jobs=-1)
hardIdx = prediction != y
linearBaseline = linear_model.LogisticRegression()
prediction = model_selection.cross_val_predict(linearBaseline, X, y, cv=10)
hardIdx |= prediction != y
randomIdx = np.random.choice([True, False], y.shape, p=[0.4, 0.6])
idx = hardIdx | randomIdx

Xhard, yhard = X[idx], y[idx]


# %%
def plotRangeScore(ran, gridSearch):
  plt.plot(ran, gridSearch.cv_results_['mean_test_score'])

# %%
# n-estimators: 110
estimatorRange = list(range(1, 2, 20)) + list(range(20, 200, 10))
paramSearch1 = {'n_estimators': estimatorRange}
gridSearch1 = model_selection.GridSearchCV(ensemble.GradientBoostingClassifier(learning_rate=0.2),
                                           cv=5, scoring='f1', param_grid=paramSearch1, n_jobs=-1, verbose=1)
gridSearch1.fit(X, y)

# %%
# learning rate: 0.21
lrRange = np.arange(0.01, 1.01, 0.05)
paramSearch2 = {'learning_rate': lrRange}
gridSearch2 = model_selection.GridSearchCV(ensemble.GradientBoostingClassifier(n_estimators=20),
                                           cv=5, scoring='f1', param_grid=paramSearch2, n_jobs=-1, verbose=1)
gridSearch2.fit(X, y)

# %%
# tree depth: 4; min_samples_split: 196
paramSearch3 = {'max_depth':range(2,5,1), 'min_samples_split':range(190, 220, 2)}
gridSearch3 = model_selection.GridSearchCV(ensemble.GradientBoostingClassifier(n_estimators=20, learning_rate=0.2),
                                           cv=5, scoring='f1', param_grid=paramSearch3, n_jobs=-1, verbose=1)
gridSearch3.fit(X, y)

# %%
sns.heatmap(gridSearch3.cv_results_['mean_test_score'].reshape(3, 15))

# %%
# min_samples_leaf: 125
paramSearch4 = {'min_samples_leaf': range(1,200,2)}
gridSearch4 = model_selection.GridSearchCV(ensemble.GradientBoostingClassifier(
  n_estimators=20, learning_rate=0.2, max_depth=3, min_samples_split=196),
  cv=5, scoring='f1', param_grid=paramSearch4, n_jobs=-1, verbose=1)
gridSearch4.fit(X, y)

# %%
# subsample: 0.57
paramSearch5 = {'subsample': np.arange(0.55, 0.85, 0.01)}
gridSearch5 = model_selection.GridSearchCV(ensemble.GradientBoostingClassifier(
  n_estimators=110, learning_rate=0.2, max_depth=3, min_samples_leaf=125, min_samples_split=196),
  cv=5, scoring='f1', param_grid=paramSearch5, n_jobs=-1, verbose=1)
gridSearch5.fit(X, y)

# %%
bestModel = ensemble.GradientBoostingClassifier(
  n_estimators=110, learning_rate=0.2, max_depth=4, min_samples_leaf=125, min_samples_split=196, subsample=0.57)
basicModel = ensemble.GradientBoostingClassifier()
bestModelScores = model_selection.cross_val_score(bestModel, X, y, cv=20, n_jobs=-1)
basicModelScores = model_selection.cross_val_score(basicModel, X, y, cv=20, n_jobs=-1)

# %%
plt.plot(bestModelScores, label='best')
plt.plot(basicModelScores, label='basic')
plt.legend()

# %%
bestModel = ensemble.GradientBoostingClassifier(
  n_estimators=110, learning_rate=0.2, max_depth=3, min_samples_leaf=125, min_samples_split=196, subsample=0.57)
bestModel.fit(X, y)

testSetNoNA = testSet.dropna()
Xtest, ytest = testSetNoNA.drop('flag', axis=1), testSetNoNA.flag
Xtest = scaler.transform(Xtest)
metrics.plot_confusion_matrix(bestModel, Xtest, ytest)

# %%
# predict training set labels
joblib.dump(bestModel, 'models/GBDT-from-validation.pkl')

Xtest = scaler.transform(testSet.drop('flag', axis=1))
inferLabels = pd.DataFrame({'ID': testSet.index, 'flag': bestModel.predict(Xtest)})

# %%
inferLabels.to_csv('dataset/infer-train-flag.csv', index=False)

# %%
