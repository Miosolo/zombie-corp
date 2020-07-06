# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn import cluster
from sklearn import preprocessing
from sklearn import feature_selection

# %%
def mypairplot(df, flag, dots):
  try:
    df = df.set_index('ID')
  except:
    pass
  
  df.iloc[:, :] = preprocessing.scale(df.to_numpy())
  df = pd.merge(df, flag, left_index=True, right_on='ID').drop('ID', axis=1).sample(dots)
  try:
    df = df.drop('year', axis=1)
  except:
    pass
  g = sns.PairGrid(df, hue='flag')
  g.map_lower(sns.kdeplot, shade=True, shade_lowest=False)
  g.map_diag(plt.hist)
  g.map_upper(plt.scatter)
  g.add_legend()


def addTimeSeriesFeatures(df: pd.DataFrame):

  baseline = df.drop('year', axis=1).groupby('ID').mean()
  baseline.columns = [c + '_mean' for c in baseline.columns]

  def getYearData(yr): return df[df.year == yr].set_index('ID').drop('year', axis=1)

  # get year diff
  d2015, d2016, d2017 = (getYearData(y) for y in (2015, 2016, 2017))
  diffmean = d2017 - d2015
  diffmean.columns = [c + '_d1_mean' for c in diffmean.columns]
  diffdiff = d2017 + d2015 - 2 * d2016
  diffdiff.columns = [c + '_d2' for c in diffdiff.columns]

  return pd.concat([baseline, diffmean, diffdiff], axis=1)


def fillYear(df):
  yearPad = df.year.fillna('pad')
  yearGrp = df.groupby(yearPad)

  commonID = np.intersect1d(yearGrp.get_group(2015).ID,
                            yearGrp.get_group(2016).ID)
  commonID = np.intersect1d(yearGrp.get_group(2017).ID, commonID)

  wrongID = np.setdiff1d(df.ID, commonID)
  for i in wrongID:
    years = df.year[df.ID==i]
    missingYear = set([2015., 2016., 2017.]) - set(years)
    missingYearIdx = years[years.isna()].index
    for idx in missingYearIdx:
      df.at[idx, 'year'] = missingYear.pop()
  
  df.loc[df.year.isna(), 'year'] = yearPad[df.year.isna()]
  return df


# add slope features to year report
def addSlopeFeatures(reportdf):
  ratePairs = {
    '资产总额': ['负债总额', '营业总收入'],
    '营业总收入': ['主营业务收入', '利润总额', '净利润', '纳税总额']
  }
  for x in ratePairs.keys():
    for y in ratePairs[x]:
      bottomIdx = reportdf[y].idxmin()
      xmin, ymin = reportdf[x][bottomIdx], reportdf[y][bottomIdx]
      slope = (reportdf[y] - ymin) / (reportdf[x] - xmin)
      yna = reportdf[y].isna()
      reportdf.loc[yna, y] = slope.median() * (reportdf.loc[yna, x] - reportdf[x].min()) + reportdf[y].min()
      angle = np.arctan(slope.fillna(slope.median()))
      reportdf[y+'/'+x] = angle
  
  return reportdf

# %%
if __name__ == "__main__":
  # sns.set(font='Microsoft Yahei')

  # set data files
  datasetConfig = {
    'train': {
      'base': 'base_train_sum.csv',
      'finance': 'money_report_train_sum.csv',
      'patent': 'patent_train_sum.csv',
      'report': 'year_report_train_sum.csv'
    },
    'validate': {
      'base': 'base_verify1.csv',
      'finance': 'money_information_verify1.csv',
      'patent': 'patent_information_verify1.csv',
      'report': 'year_report_verify1.csv'
    },
    'validate2': {
      'base': 'base_verify2_sum.csv',
      'finance': 'money_report_verify2_sum.csv',
      'patent': 'knowledge_verify2_sum.csv',
      'report': 'year_report_verify2_sum.csv'
    },
    'test': {
      'base': 'base_test_sum.csv',
      'finance': 'money_report_test_sum.csv',
      'patent': 'knowledge_test_sum.csv',
      'report': 'year_report_test_sum.csv'
    }
  }
  MODE = 'validate'

  # %%
  basedf, financedf, patentdf, reportdf = (pd.read_csv(os.path.join(
    'dataset', datasetConfig[MODE][sub])) for sub in datasetConfig['train'].keys())
  if MODE != 'test':
    flagdf = basedf[['ID', 'flag']]
    basedf = basedf.drop('flag', axis=1)
  basedf0, financedf0, patentdf0, reportdf0 = basedf.copy(), financedf.copy(), patentdf.copy(), reportdf.copy()

  # %%
  # fill NA in basedf
  basedfNumeric = basedf[['ID', '注册时间', '注册资本', '控制人持股比例']]
  basedfNumeric = basedfNumeric.fillna(basedfNumeric.mean())
  # basedf[['行业', '区域', '企业类型', '控制人类型']] = basedf[['行业', '区域', '企业类型', '控制人类型']].fillna('NA')
  basedf = pd.concat([basedfNumeric,
                      pd.get_dummies(basedf.行业, prefix='行业'),
                      pd.get_dummies(basedf.区域, prefix='区域'),
                      pd.get_dummies(basedf.企业类型, prefix='企业类型'),
                      pd.get_dummies(basedf.控制人类型, prefix='控制人类型'), ], axis=1)

  # %%
  # fill NA in patentdf
  # patentdf = patentdf.fillna('NA')
  patentdf.iloc[:, 1:] = patentdf.iloc[:, 1:].astype('float')
  patentdf = pd.concat([patentdf.ID,
                        pd.get_dummies(patentdf.专利, prefix='专利'),
                        pd.get_dummies(patentdf.商标, prefix='商标'),
                        pd.get_dummies(patentdf.著作权, prefix='著作权')], axis=1)


  # %%
  # fill NA in financedf
  # filling with quota-rate relationship 存在先验假设
  financeCategories = {'债权融资': 0.08, '股权融资': 0.04,
                      '内部融资和贸易融资': 0.06, '项目融资和政策融资': 0.06}
  for c, rate in financeCategories.items():
    quota = c+'额度'
    cost = c+'成本'
    quotaNull = financedf[quota].isnull() & financedf[cost].notnull()
    financedf.loc[quotaNull, quota] = financedf.loc[quotaNull, cost] / rate
    # drop duplicates
    financedf = financedf.drop(cost, axis=1)

  # other numeric values: use median = 0
  # log-like transformation
  financedf.iloc[:, 2:] = financedf.iloc[:, 2:].fillna(0).applymap(np.log1p)
  # fill NA in year by padding
  financedf = fillYear(financedf)
  financedf = addTimeSeriesFeatures(financedf)

  # %%
  # fill NA in reportdf
  # discard surplus, duplicate
  reportdf = reportdf.drop('所有者权益合计', axis=1)

  # filling personnel by mean, since irrelavance
  reportdf.从业人数 = reportdf.从业人数.fillna(reportdf.从业人数.mean())
  reportdf.资产总额 = reportdf.资产总额.fillna(reportdf.资产总额.median())

  # replace highly-dependent vars with rate
  reportdf = addSlopeFeatures(reportdf)
  # reportdf = reportdf.drop(np.sum(list(ratePairs.values())), axis=1)
  # padding year
  # reportdf = fillYear(reportdf)
  # reportdf = addTimeSeriesFeatures(reportdf)

  # %%
  # store results
  basedf.to_hdf('dataset/preprocessed-data.h5', key='base_'+MODE)
  patentdf.to_hdf('dataset/preprocessed-data.h5', key='patent_'+MODE)
  financedf.to_hdf('dataset/preprocessed-data.h5', key='finance_'+MODE)
  reportdf.to_hdf('dataset/preprocessed-data.h5', key='report_'+MODE)
  if MODE != 'test':
    flagdf.to_hdf('dataset/preprocessed-data.h5', key='flag_'+MODE)

  # %%
  aiodf = basedf
  for df in (patentdf, financedf, reportdf):
    aiodf = pd.merge(aiodf, df, on='ID')
  aiodf.to_hdf('dataset/preprocessed-data.h5', key='all_'+MODE)
