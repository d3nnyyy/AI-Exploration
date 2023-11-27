import en_core_web_sm
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("Fake_Real_Data.csv")
print(df.shape)
print(df.head(5))

print(df.label.value_counts())

df['label_num'] = df['label'].map({'Fake': 0, 'Real': 1})
print(df.head(5))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df.Text,
    df.label_num,
    test_size=0.2,
    random_state=2022,
    stratify=df.label_num
)

print(X_train.shape, X_test.shape)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# clf = Pipeline([
#     ('vectorizer_trigrams', CountVectorizer(ngram_range=(1, 3))),
#     ('KNN', (KNeighborsClassifier(n_neighbors=10, metric='euclidean')))
# ])
#
# clf.fit(X_train, y_train)
#
# y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))
#
# clf = Pipeline([
#     ('vectorizer_trigrams', CountVectorizer(ngram_range=(1, 3))),
#     ('KNN', (KNeighborsClassifier(n_neighbors=10, metric='cosine')))
# ])
#
# clf.fit(X_train, y_train)
#
# y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))
#
# clf = Pipeline([
#     ('vectorizer_trigrams', CountVectorizer(ngram_range=(1, 3))),
#     ('RFC', (RandomForestClassifier()))
# ])
#
# clf.fit(X_train, y_train)
#
# y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))

clf = Pipeline([
    ('vectorizer_trigrams', CountVectorizer(ngram_range=(1, 3))),
    ('MNB', (MultinomialNB(alpha=0.75)))
])

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

import spacy

nlp = en_core_web_sm.load()


def preprocess(text):
    # remove stop words and lemmatize the text
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)

    return " ".join(filtered_tokens)


df['preprocessed_txt'] = df['Text'].apply(preprocess)
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(
    df.preprocessed_txt,
    df.label_num,
    test_size=0.2,  # 20% samples will go to test dataset
    random_state=2022,
    stratify=df.label_num
)

clf = Pipeline([
    ('vectorizer_n_grams', CountVectorizer(ngram_range=(3, 3))),  # using the ngram_range parameter
    ('random_forest', (RandomForestClassifier()))
])

# 2. fit with X_train and y_train
clf.fit(X_train, y_train)

# 3. get the predictions for X_test and store it in y_pred
y_pred = clf.predict(X_test)

# 4. print the classfication report
print(classification_report(y_test, y_pred))
