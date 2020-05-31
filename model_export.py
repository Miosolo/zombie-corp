import pandas as pd
from sklearn.externals import joblib
from sklearn import metrics

from data_preprocessing import addTimeSeriesFeatures, addSlopeFeatures

class ExportedModel():
  def __init__(self, modelPath: str):
    self.model, self.featureColumns = joblib.load(modelPath)

  def predict(self, yearReport: pd.DataFrame):
    df = addSlopeFeatures(yearReport)
    df = addTimeSeriesFeatures(df) # indexed by 'year'
    feat = df[self.featureColumns]
    inference = pd.Series(self.model.predict(feat), index=df.index)
    
    return inference.to_dict()
  
  def explain(self, feat):
    pass


if __name__ == "__main__":
  model = ExportedModel(modelPath='models/GBDT-pipe-from-validation.pkl')
  df = pd.read_csv('dataset/year_report_test_sum.csv')
  inf = model.predict(df)
  lab = pd.read_csv('results/infer-test-flag-from-validation.csv')
  print(metrics.confusion_matrix(lab.flag, list(inf.keys())))