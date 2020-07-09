# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn import cluster, manifold, neighbors, linear_model, ensemble, neural_network, tree, svm
from sklearn import metrics, model_selection, preprocessing, feature_selection
from sklearn.pipeline import Pipeline
# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler

# Decision Tree Viz
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.externals.six import StringIO
import pydotplus
import graphviz
from IPython.display import SVG

import pickle

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

def decisionTreeViz(model, fileName=None):
  dot_data = StringIO()
  export_graphviz(model, feature_names=features, class_names=['正常企业', '僵尸企业'],
                  out_file=dot_data, filled=True, rounded=True, special_characters=True)
  graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
  if fileName:
    with open(os.path.join('figures', fileName), 'wb') as f:
      f.write(graph.create_svg())
  return SVG(graph.create_svg())

def tryAllClassifers(Xtrain, Xtest, ytrain, ytest):
  # try with different classifiers
  _, axes = plt.subplots(3, 3, figsize=(14, 14))
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

def tryClassifiersCV(X, y, models=None, cv=5):
  # cross validation
  if models is None:
    models = [tree.DecisionTreeClassifier(),
              tree.DecisionTreeClassifier(min_samples_split=4, max_depth=3, min_samples_leaf=4, 
                splitter='best', min_impurity_decrease=0., ),
              ensemble.RandomForestClassifier(),
              ensemble.AdaBoostClassifier(),
              ensemble.GradientBoostingClassifier()]
    labels = ['DecisionTreeClassifier',
              'DecisionTreeClassifier(tuned)',
              'RandomForestClassifier',
              'AdaBoostClassifier',
              'GradientBoostingClassifier']
  scores = {}
  for model, name in zip(models, labels):
    score = model_selection.cross_val_score(model, X, y, cv=cv, n_jobs=-1)
    scores[name] = score
    plt.plot(score, label=name)
  plt.legend()

  return scores

def paramSearch2d(ps: dict, X, y, computed=None):
  if len(ps) != 2:
    raise RuntimeError()

  if computed:
    gs = computed
  else:
    gs = model_selection.GridSearchCV(tree.DecisionTreeClassifier(),
      cv=5, scoring='f1', param_grid=ps, n_jobs=-1, verbose=1)
    gs.fit(X, y)

  k1, k2 = gs.cv_results_['params'][0].keys()
  l1, l2 = len(ps[k1]), len(ps[k2])

  plt.figure(figsize=sorted([l1/3, l2/3,], reverse=True))

  if l1 > l2:
    k1, k2 = k2, k1
    l1, l2 = l2, l1
    data = gs.cv_results_['mean_test_score'].reshape((l2, l1)).T
  else:
    data=gs.cv_results_['mean_test_score'].reshape((l1, l2))

  df = pd.DataFrame(data=data, index=map(lambda o: str(o), ps[k1]), columns=map(lambda o: str(o), ps[k2]))
  
  g = sns.heatmap(df)
  g.set(ylabel=k1, xlabel=k2)

  return gs

# %%
# validate -> train
trainFeatures = pd.read_hdf('dataset/preprocessed-data.h5', key='all_validate')
trainFeatures = pd.concat([trainFeatures,
                           pd.read_hdf('dataset/preprocessed-data.h5', key='all_validate2')])
trainLabels = pd.read_hdf('dataset/preprocessed-data.h5', key='flag_validate')
trainLabels = pd.concat([trainLabels, 
                         pd.read_hdf('dataset/preprocessed-data.h5', key='flag_validate2')])

testFeatures = pd.read_hdf('dataset/preprocessed-data.h5', key='all_train')
testLabels = pd.read_hdf('dataset/preprocessed-data.h5', key='flag_train')

trainFeatures = pd.concat([trainFeatures, testFeatures])
trainLabels = pd.concat([trainLabels, testLabels])

# # train -> validate
# trainFeatures = pd.read_hdf('dataset/preprocessed-data.h5', key='all_train')
# trainLabels = pd.read_csv('dataset/infer-train-flag.csv')
# testFeatures = pd.read_hdf('dataset/preprocessed-data.h5', key='all_validate')
# testLabels = pd.read_hdf('dataset/preprocessed-data.h5', key='flag_validate')

inferenceFeatures = pd.read_hdf('dataset/preprocessed-data.h5', key='all_test')

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
trainFeatures = trainFeatures[['ID'] + list(features)]
testFeatures = testFeatures[['ID'] + list(features)]
inferenceFeatures = inferenceFeatures[['ID'] + list(features)]

trainSet = pd.merge(trainFeatures, trainLabels, how='inner', on='ID')
trainSet = trainSet.dropna().set_index('ID')
testSet = pd.merge(testFeatures, testLabels, how='inner', on='ID')
testSet = testSet.set_index('ID')
inferenceFeatures = inferenceFeatures.set_index('ID')

features = trainSet.columns[:-1]

# %%
# prepare data
# trainSet = trainSet.sample(TRAIN_SAMPLES)
X, y = trainSet.drop('flag', axis=1), trainSet.flag
# X, y = RandomUnderSampler().fit_resample(X, y)
# X, y = SMOTE().fit_resample(X, y)

# %%
# rough test
pipe = Pipeline([ # ('scaler', preprocessing.StandardScaler()),
                 ('DecisionTree', tree.DecisionTreeClassifier(min_samples_split=4, max_depth=3, 
                                                              min_samples_leaf=4, splitter='best', 
                                                              min_impurity_decrease=0., ))
                ])
# cross-validation
selfScores = model_selection.cross_val_score(pipe, X, y, cv=20, n_jobs=-1)
plt.plot(selfScores)

# %%
# on test set
pipe.fit(X, y)

testSet = testSet.dropna()
Xtest, ytest = testSet.drop('flag', axis=1), testSet.flag
metrics.plot_confusion_matrix(pipe, Xtest, ytest)
metrics.plot_roc_curve(pipe, Xtest, ytest)

# %%
with open('models/CART-from-validation.pkl', 'wb') as f:
  pickle.dump(pipe, f)

inference = pipe.predict(inferenceFeatures)
inferenceFlag = pd.DataFrame({'ID': inferenceFeatures.index, 'flag': inference})
inferenceFlag.flag = inferenceFlag.flag.astype('int')
inferenceFlag.to_csv('results/infer-test-flag-from-validation.csv', index=False)

# %%
# set testing features
X = preprocessing.StandardScaler().fit_transform(X)

# param searching for Decision Tree
# %%
paramSearch1 = {'max_depth':range(2, 20, 1), 'min_samples_split':range(1, 50, 1)}
gridSearch1 = paramSearch2d(paramSearch1, X, y)
# => min_samples_split <= 5

# %%
paramSearch2 = {'min_samples_leaf': range(1,20,1), 'max_depth':range(2, 20, 1)}
gridSearch2 = paramSearch2d(paramSearch2, X, y)
# min_samples_leaf <= 3 & max_depth < 11

# %%
paramSearch3 = {'max_depth':range(2, 20, 1), 'splitter': ['best', 'random']}
gridSearch3 = paramSearch2d(paramSearch3, X, y)
# => best

# %%
paramSearch4 = {'max_leaf_nodes': range(1, 100, 2), 'max_depth':range(2, 20, 1)}
gridSearch4 = paramSearch2d(paramSearch4, X, y)
# => max_depth <=4

# %%
paramSearch5 = {'class_weight': list(map(lambda f: {0: 1-f, 1: f}, np.arange(0, 1, 0.05))), 'max_depth':range(2, 10, 1)}
gridSearch5 = paramSearch2d(paramSearch5, X, y)
# => max_depth > 2

# %%
paramSearch6 = {'min_impurity_decrease': np.arange(0, 0.02, 0.001), 'max_depth':range(2, 10, 1)}
gridSearch6 = paramSearch2d(paramSearch6, X, y)
# => min_impurity_decrease = 0

# %%
bestModel = tree.DecisionTreeClassifier(min_samples_split=4, max_depth=3, min_samples_leaf=4, 
  splitter='best', min_impurity_decrease=0., )
basicModel = tree.DecisionTreeClassifier()
bestModelScores = model_selection.cross_val_score(bestModel, X, y, cv=20, n_jobs=-1)
basicModelScores = model_selection.cross_val_score(basicModel, X, y, cv=20, n_jobs=-1)
plt.plot(bestModelScores, label='tuned')
plt.plot(basicModelScores, label='default')
plt.legend()
