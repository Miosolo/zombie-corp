# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cluster, manifold, neighbors
from sklearn import metrics, model_selection, preprocessing

from tqdm import tqdm

# %%
corpTrain = pd.read_hdf('dataset/preprocessed-data.h5', key='corp_train')
corpVal = pd.read_hdf('dataset/preprocessed-data.h5', key='corp_validate')
corpUnion = pd.concat([corpTrain, corpVal])

moneyTrain = pd.read_hdf('dataset/preprocessed-data.h5', key='money_train')
moneyVal = pd.read_hdf('dataset/preprocessed-data.h5', key='money_validate')
moneyUnion = pd.concat([moneyTrain, moneyVal])

flagVal = pd.read_hdf('dataset/preprocessed-data.h5', key='flag_validate')
baseVal = pd.read_hdf('dataset/preprocessed-data.h5', key='base_validate')
financeVal = pd.read_hdf('dataset/preprocessed-data.h5', key='finance_validate')
reportVal = pd.read_hdf('dataset/preprocessed-data.h5', key='report_validate')
patentVal = pd.read_hdf('dataset/preprocessed-data.h5', key='patent_validate')

# %%
# dataset = corpVal.drop('ID', axis=1).dropna().sample(frac=0.2)
dataset = pd.merge(financeVal, flagVal, how='inner', on='ID')
baseline = dataset.groupby('ID').mean()

# %%
# self diff
selectedID = np.random.choice(dataset.ID, size=500, replace=False)
smalldf = dataset[dataset.ID.isin(selectedID)]
idGrp = smalldf.groupby('ID')

# plot year diff
canvas, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, subdf in idGrp:
  for col, ax in zip(subdf.iloc[:, 2:-1], axes.flatten()):
    ax.plot(preprocessing.minmax_scale(subdf[col]))
for ax, attr in zip(axes.flatten(), dataset.columns[2:-1]):
  ax.set(title=attr)

# %%
# diff statisitcs
diff1 = idGrp.agg(lambda d: (d.iloc[1] - d.iloc[0])/d.sum())
diff2 = idGrp.agg(lambda d: (d.iloc[2] - d.iloc[1])/d.sum())

canvas, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, attr in zip(axes.flatten(), dataset.columns[2:-1]):
  # ax.hist(diff1[attr], bins=30, alpha=0.6)
  # ax.hist(diff2[attr], bins=30, color='g', alpha=0.6)
  ax.hist(diff2[attr] - diff1[attr], bins=30, color='orange', alpha=0.4)
  ax.set(title=attr)