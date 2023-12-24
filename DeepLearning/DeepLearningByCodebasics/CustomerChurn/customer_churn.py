import keras
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sympy.printing.tensorflow import tensorflow

df = pd.read_csv("customer_churn.csv")

df.drop('customerID', axis='columns', inplace=True)

df1 = df[df.TotalCharges != ' ']
df1.TotalCharges = pd.to_numeric(df1.TotalCharges)


# print(df1.TotalCharges.values)
# print(df.dtypes)
# print(df.head())


def print_unique(df):
    for column in df:
        if df[column].dtypes == 'object':
            print(f'{column}: {df[column].unique()}')


# print_unique(df1)

df1.replace('No internet service', 'No', inplace=True)
df1.replace('No phone service', 'No', inplace=True)

# print_unique(df1)

yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']

for col in yes_no_columns:
    df1[col].replace({'Yes': 1, 'No': 0}, inplace=True)

# for col in df1:
# print(f'{col}: {df1[col].unique()}')

# print_unique(df1)

df1['gender'].replace({'Female': 1, 'Male': 0}, inplace=True)
# print(df1.gender.unique())
# print(df1.columns)

pd.set_option('display.max_columns', None)

df2 = pd.get_dummies(data=df1, columns=['InternetService', 'Contract', 'PaymentMethod'])

columns_to_convert = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
                      'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                      'StreamingMovies', 'PaperlessBilling', 'Churn', 'InternetService_DSL',
                      'InternetService_Fiber optic', 'InternetService_No', 'Contract_Month-to-month',
                      'Contract_One year', 'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)',
                      'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
                      'PaymentMethod_Mailed check']
df2[columns_to_convert] = df2[columns_to_convert].astype(int)

print(df2.columns)
print(df2.sample(5))
print(df2.dtypes)

cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']

scaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])

x = df2.drop('Churn', axis='columns')
y = df2['Churn']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=100)
model.evaluate(x_test, y_test)
yp = model.predict(x_test)

y_pred = []
for el in yp:
    if el > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

print(y_pred[:10])
print(y_test[:10])

cm = tensorflow.math.confusion_matrix(labels=y_test,predictions=y_pred)

plt.figure(figsize = (10,7))
seaborn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()