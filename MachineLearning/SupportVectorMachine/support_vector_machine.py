import pandas as pd
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

df['target'] = iris.target

print(df[df.target == 2].head())

df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])

print(df.head)

df0 = df[df.target == 0]
df1 = df[df.target == 1]
df2 = df[df.target == 2]

plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color='green', marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color='red', marker='+')
plt.show()

X = df.drop(['target', 'flower_name'],axis='columns')
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = SVC(kernel='linear')

model.fit(X_train, y_train)
print(model.score(X_test, y_test))
