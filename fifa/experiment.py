import pandas as pd
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC

def estimate(df, target, model, score, splits=10):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    df = df.select_dtypes(include=numerics)
    print(df.columns)
    df = df.dropna()
    X = df.drop(target, axis=1).values
    Y = df[target].values

    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('m', model))
    model = Pipeline(estimators)

    kfold = ShuffleSplit(n_splits=splits, random_state=1, test_size=0.1)
    results = cross_val_score(model, X, Y, cv=kfold, scoring=score)
    return results.mean()



df_learn = pd.read_csv('learningData.csv')
df_stats = pd.read_csv('stats.csv')
df = df_learn.set_index('GameID').join(df_stats.set_index('GameID'))

r1 = estimate(df, 'Man of the Match', SVC(), 'f1')
r2 = estimate(df_learn, 'Man of the Match', SVC(), 'f1')

print('neg-RSME (higher better):')
print('augmented score : ', r1)
print('base score : ', r2)

print(len(df), len(df_learn))