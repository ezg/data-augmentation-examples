import pandas as pd
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

def estimate(df, target, model, score):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    df = df.select_dtypes(include=numerics)
    df = df.dropna()
    X = df.drop(target, axis=1).values
    Y = df[target].values

    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('m', model))
    model = Pipeline(estimators)

    kfold = KFold(n_splits=10, random_state=1)
    results = cross_val_score(model, X, Y, cv=kfold, scoring=score)
    return results.mean()



df_pop = pd.read_csv('population.csv').drop('N', axis=1)
df_pov = pd.read_csv('poverty.csv')
df = df_pov.set_index('FIPS').join(df_pop.set_index('FIPS'))

r1 = estimate(df, 'POVALL_2016', RandomForestRegressor(), 'neg_mean_absolute_error')
r2 = estimate(df_pov, 'POVALL_2016', RandomForestRegressor(), 'neg_mean_absolute_error')

print('neg-RSME (higher better):')
print('augmented score : ', r1)
print('base score : ', r2)