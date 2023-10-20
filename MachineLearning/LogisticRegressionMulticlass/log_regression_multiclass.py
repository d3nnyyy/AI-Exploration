from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

digits = load_digits()
# print(digits.data[0])

plt.gray()
for i in range(5):
    plt.matshow(digits.images[i])

# plt.show()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))


print(model.predict([digits.data[67]]))
print(model.predict(digits.data[0:5]))

y_predicted = model.predict(X_test)

cm = confusion_matrix(y_test, y_predicted)
print(cm)