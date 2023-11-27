import en_core_web_sm
import spacy

nlp = en_core_web_sm.load()

with open("news_story.txt", "r") as f:
    news_text = f.read()

doc = nlp(news_text)

numeral_tokens = []
noun_tokens = []

for token in doc:
    if token.pos_ == "NOUN":
        noun_tokens.append(token)
    elif token.pos_ == 'NUM':
        numeral_tokens.append(token)

print(numeral_tokens[:10])

count = doc.count_by(spacy.attrs.POS)
print(count)

for k, v in count.items():
    print(doc.vocab[k].text, "|", v)
