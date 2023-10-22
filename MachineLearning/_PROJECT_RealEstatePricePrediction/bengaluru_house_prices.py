import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams["figure.figsize"] = (20, 10)

df1 = pd.read_csv("bengaluru_house_prices.csv")
df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
df3 = df2.dropna()

print(df3.isnull().sum())
print(df3['size'].unique())

df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


print(df3[~df3['total_sqft'].apply(is_float)].head(10))


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return float(tokens[0]) + float(tokens[1]) / 2
    try:
        return float(x)
    except:
        return None


df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
print(df4.head())

df5 = df4.copy()
df5['price_per_sqft'] = df5['price'] * 100000 / df5['total_sqft']
print(df5.head())

print(len(df5.location.unique()))

df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
print(location_stats)

print(len(location_stats[location_stats <= 10]))

location_stats_less_than_10 = location_stats[location_stats <= 10]

df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

print(len(df5.location.unique()))