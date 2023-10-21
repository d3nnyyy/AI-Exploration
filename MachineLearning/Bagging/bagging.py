import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

df = pd.read_csv("diabetes.csv")

x = df.drop('Outcome', axis='columns')
y = df.Outcome

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, stratify=y, random_state=10)

model = DecisionTreeClassifier()
scores = cross_val_score(DecisionTreeClassifier(), x, y, cv=5)

bag_model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0
)

# scores = cross_val_score(bag_model, x, y, cv=5)
# print(scores.mean())

bag_model.fit(x_train, y_train)
print(bag_model.oob_score_)
print(bag_model.score(x_test, y_test))