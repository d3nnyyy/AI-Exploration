import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

dataset = load_digits()
print(dataset.keys())
print(dataset.data[3].reshape(8,8))

plt.gray()
plt.matshow(dataset.data[3].reshape(8,8))

# plt.show()

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

x = df
y = dataset.target

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
print(x_scaled)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
print(log_reg.score(X_test, y_test))

pca = PCA(0.95)
x_pca = pca.fit_transform(x)

X_train_pca, X_test_pca, y_train, y_test = train_test_split(x_pca, y, test_size=0.2)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_pca, y_train)
print(log_reg.score(X_test_pca, y_test))