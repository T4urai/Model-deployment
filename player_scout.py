import pandas as pd

from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
import statsmodels.formula.api as smf
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import scipy.spatial
import sys

def scout(player_statistic):
    df=pd.read_excel("Projectdata.xlsx")
    x1=df.drop(['Players','Clubs','Sub','Pen M','Sub','OG','PlayerCategory'],axis=1)
    nan_value=float("NaN")
    x1.replace('-',0,inplace=True)
    y1=df['PlayerCategory']

    labelencoder_x1 = LabelEncoder()
    x1['Position'] = labelencoder_x1.fit_transform(x1['Position'])
    y1=labelencoder_x1.fit_transform(y1)
    
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=.2, random_state=41)

    clf=RandomForestClassifier(n_jobs=2,random_state=0)
    clf.fit(x_train,y_train)
    preds=df.PlayerCategory[clf.predict(x_test)]
    
    return preds

      
    