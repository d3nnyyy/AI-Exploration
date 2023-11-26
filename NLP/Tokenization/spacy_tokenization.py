import spacy

nlp = spacy.blank("en")

text = '''
Look for data to help you address the question. Governments are good
sources because data from public research is often freely available. Good
places to start include http://www.data.gov/, and http://www.science.
gov/, and in the United Kingdom, http://data.gov.uk/.
Two of my favorite data sets are the General Social Survey at http://www3.norc.org/gss+website/, 
and the European Social Survey at http://www.europeansocialsurvey.org/.
'''

doc = nlp(text)
data_websites = [token.text for token in doc if token.like_url]
print(data_websites)

transactions = "Tony gave two $ to Peter, Bruce gave 500 € to Steve"

doc = nlp(transactions)
data_amount_currency = [[token, doc[token.i + 1]] for token in doc if token.like_num and doc[token.i + 1].is_currency]

print(data_amount_currency)
