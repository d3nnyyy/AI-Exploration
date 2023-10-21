import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from custom_messages import spam_message, ham_message

df = pd.read_csv("spam.csv")

# print(df.groupby('Category').describe())

df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

print(df.head())

X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size=0.25)

clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

clf.fit(X_train, y_train)

emails = [
    spam_message, ham_message
]

print(clf.predict(emails))