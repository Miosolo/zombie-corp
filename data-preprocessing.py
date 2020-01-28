# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
from matplotlib.font_manager import FontProperties
myfont=FontProperties(fname=r'C:\Windows\Fonts\msyh.ttc',size=10)
sns.set(font=myfont.get_name())

# %%
basedf = pd.read_csv('dataset/base_train_sum.csv')
moneydf = pd.read_csv('dataset/money_report_train_sum.csv')
patentdf = pd.read_csv('dataset/patent_train_sum.csv')
reportdf = pd.read_csv('dataset/year_report_train_sum.csv')

# %%
# fill NA in basedf
basedfNumric = basedf[['ID', '注册时间', '注册资本', '控制人持股比例']]
basedfNumric = basedfNumric.fillna(basedfNumric.mean())
basedf = pd.concat([basedfNumric,
                    pd.get_dummies(basedf['行业'], prefix='行业'),
                    pd.get_dummies(basedf['区域'], prefix='区域'),
                    pd.get_dummies(basedf['企业类型'], prefix='企业类型'),
                    pd.get_dummies(basedf['控制人类型'], prefix='控制人类型'),
                    basedf['flag']], axis=1)

# %%
# fill NA in patentdf
patentdf = pd.concat([patentdf['ID'], 
                      pd.get_dummies(patentdf['专利'], prefix='专利'), 
                      pd.get_dummies(patentdf['商标'], prefix='商标'), 
                      pd.get_dummies(patentdf['著作权'], prefix='著作权')], axis=1)

# %%
# fill NA in moneydf
# filling with quota-rate relationship
moneyCategories = {'债权融资': 0.08, '股权融资': 0.04, '内部融资和贸易融资': 0.06, '项目融资和政策融资': 0.06}
for c, rate in moneyCategories.items():
  quota = c+'额度'
  cost = c+'成本'
  quotaNull = moneydf[quota].isnull() & moneydf[cost].notnull()
  costNull = moneydf[quota].notnull() & moneydf[cost].isnull()
  moneydf.loc[quotaNull, quota] = moneydf.loc[quotaNull, cost] / rate
  moneydf.loc[costNull, cost] = moneydf.loc[costNull, quota] * rate
# other numeric values: use mode = 0
moneydf.iloc[:, 2:] = moneydf.iloc[:, 2:].fillna(0)
# fill NA in year by padding
moneydf.year = moneydf.year.fillna(method='pad') 

# %%
# fill NA in reportdf
# discard surplus, duplicate
reportdf = reportdf.drop('所有者权益合计', axis=1)
# padding year
reportdf.year = reportdf.year.fillna(method='pad')
# filling personnel by mean, since irrelavance
reportdf['从业人数'] = reportdf['从业人数'].fillna(reportdf['从业人数'].mean())
reportdf['资产总额'] = reportdf['资产总额'].fillna(reportdf['资产总额'].median())
# make use of correlation rates
debtCapitalRate = reportdf['负债总额'] / reportdf['资产总额']
reportdf['负债总额'] = debtCapitalRate.fillna(debtCapitalRate.median()) * reportdf['资产总额']
incomeCapitalRate = reportdf['营业总收入'] / reportdf['资产总额']
reportdf['营业总收入'] = incomeCapitalRate.fillna(incomeCapitalRate.mean()) * reportdf['资产总额']
othersIncomeRate = reportdf.iloc[:, 6:].div(reportdf['营业总收入'], axis=0) # 主营业务收入, ...
reportdf.iloc[:, 6:] = othersIncomeRate.fillna(othersIncomeRate.median()).mul(reportdf['营业总收入'], axis=0)

# %%
# store results
basedf.to_hdf('dataset/preprocessed-data.h5', key='base_train')
patentdf.to_hdf('dataset/preprocessed-data.h5', key='patent_train')
moneydf.to_hdf('dataset/preprocessed-data.h5', key='money_train')
reportdf.to_hdf('dataset/preprocessed-data.h5', key='report_train')

# %%
corpdf = pd.merge(basedf, patentdf, how='inner', on='ID')
financedf = pd.merge(moneydf, reportdf, how='inner', on=['ID', 'year'])
corpdf.to_hdf('dataset/preprocessed-data.h5', key='corp_train')
financedf.to_hdf('dataset/preprocessed-data.h5', key='finance_train')

# %%
# feature engineering ? 
