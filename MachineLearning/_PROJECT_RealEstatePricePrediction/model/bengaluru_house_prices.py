import json
import pickle

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor

# Set a common figure size for plots
matplotlib.rcParams["figure.figsize"] = (20, 10)

# Step 1: Load and Preprocess Data
df1 = pd.read_csv("bengaluru_house_prices.csv")
df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
df3 = df2.dropna()

# Print the count of null values in each column
print(df3.isnull().sum())

# Display unique values in the 'size' column
print(df3['size'].unique())

# Extract the number of bedrooms (BHK) from the 'size' column and create a new 'bhk' column
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))


# Step 2: Handle Total Square Feet Column
# Define a function to check if a value can be converted to a float
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# Display rows with non-numeric values in the 'total_sqft' column
print(df3[~df3['total_sqft'].apply(is_float)].head(10))


# Define a function to convert 'total_sqft' values to numeric, handling ranges like '1000-1500'
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

# Step 3: Calculate Price Per Square Feet
df5 = df4.copy()
df5['price_per_sqft'] = df5['price'] * 100000 / df5['total_sqft']
print(df5.head())

# Step 4: Handle Location Data
# Display the count of unique locations
print(len(df5.location.unique()))

# Remove leading and trailing whitespace from location names
df5.location = df5.location.apply(lambda x: x.strip())

# Count the number of data points for each location
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
print(location_stats)

# Identify locations with less than or equal to 10 data points and categorize them as 'other'
print(len(location_stats[location_stats <= 10]))
location_stats_less_than_10 = location_stats[location_stats <= 10]
df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
print(len(df5.location.unique()))

# Step 5: Remove Outliers Based on Total Square Feet per Bedroom
print(df5[df5.total_sqft / df5.bhk < 300].head())
print(df5.shape)

df6 = df5[~(df5.total_sqft / df5.bhk < 300)]
print(df6.shape)

# Display statistics of the 'price_per_sqft' column
print(df6.price_per_sqft.describe())


# Step 6: Remove Outliers Using Standard Deviation
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out


df7 = remove_pps_outliers(df6)
print(df7.shape)


# Step 7: Create Scatter Plot
def plot_scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    matplotlib.rcParams['figure.figsize'] = (15, 10)
    plt.scatter(bhk2.total_sqft, bhk2.price, color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker='+', color='green', label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    plt.show()


# Example usage: plot_scatter_chart(df7, "Hebbal")

# Step 8: Remove BHK Outliers
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices,
                                            bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')


df8 = remove_bhk_outliers(df7)

# Step 9: Feature Engineering and Data Transformation
df9 = df8[df8.bath < df8.bhk + 2]
df10 = df9.drop(['size', 'price_per_sqft'], axis='columns')
print(df10.head(3))

# One-Hot Encoding for Location
dummies = pd.get_dummies(df10.location)
df11 = pd.concat([df10, dummies.drop('other', axis='columns')], axis='columns')
df12 = df11.drop('location', axis='columns')
print(df12.head(2))

# Step 10: Model Building
x = df12.drop('price', axis='columns')
y = df12.price
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)

# Evaluate the model
print(lr_clf.score(X_test, y_test))


# Step 11: Model Selection Using GridSearchCV
def find_best_model_using_gridsearchcv(x, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'positive': [True, False],
                'fit_intercept': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['mse', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(x, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])


# Example usage: find_best_model_using_gridsearchcv(x, y)

# Step 12: Model Prediction
def predict_price(location, sqft, bath, bhk, x):
    # global x
    loc_index = np.where(x.columns == location)[0][0]

    x = np.zeros(len(x.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


# Example usage: predict_price('1st Phase JP Nagar', 1000, 3, 3, x)

# Step 13: Save Model and Columns Information
with open('bengaluru_house_prices_model.pickle', 'wb') as f:
    pickle.dump(lr_clf, f)

columns = {
    'data_columns': [col.lower() for col in x.columns]
}

with open('columns.json', 'w') as f:
    f.write(json.dumps(columns))
