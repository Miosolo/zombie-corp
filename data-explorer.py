#%%
import pandas as pd
import matplotlib.pyplot as plt

# %%
basedf = pd.read_csv('dataset/base_train_sum.csv')
moneydf = pd.read_csv('dataset/money_report_train_sum.csv')
patentdf = pd.read_csv('dataset/patent_train_sum.csv')
reportdf = pd.read_csv('dataset/year_report_train_sum.csv')

# %%
corpdf = pd.merge(basedf, patentdf, how='left', on='ID')
financedf = pd.merge(moneydf, reportdf, how='inner', on=['ID', 'year'])

