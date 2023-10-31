import pandas as pd

df = pd.read_csv("customer_churn.csv")

df.drop('customerID', axis='columns', inplace=True)

df1 = df[df.TotalCharges != ' ']
df1.TotalCharges = pd.to_numeric(df1.TotalCharges)

print(df1.TotalCharges.values)
print(df.dtypes)
print(df.head())


def print_unique(df):
    for column in df:
        if df[column].dtypes == 'object':
            print(f'{column}: {df[column].unique()}')


print_unique(df1)

df1.replace('No internet service', 'No', inplace=True)
df1.replace('No phone service', 'No', inplace=True)

print_unique(df1)

yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']

for col in yes_no_columns:
    df1[col].replace({'Yes': 1, 'No': 0}, inplace=True)

for col in df1:
    print(f'{col}: {df1[col].unique()}')

print_unique(df1)

df1['gender'].replace({'Female': 1, 'Male': 0}, inplace=True)
print(df1.gender.unique())