import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

digits = load_digits()

df = pd.DataFrame(digits.data)
df['target'] = digits.target

X_train, X_test, y_train, y_test = train_test_split(df.drop(['target'], axis='columns'), digits.target, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

y_predicted = model.predict(X_test)

cm = confusion_matrix(y_test, y_predicted)

print(cm)
