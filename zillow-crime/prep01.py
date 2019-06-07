import pandas as pd
import os

def map_city(x):
    return str(x).zfill(3)

df = pd.read_csv('crime.csv')
df["FIPS"] = df["FIPS_ST"].map(str) + df["FIPS_CTY"].map(map_city)
df.to_csv(os.path.join('crime_clean.csv'), index=False)
print(df)