from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# Sample text data
text = """
The winner of the 2023 Ballon d’Or has been announced and Lionel Messi has won a record-extending eighth award. The Inter Miami and Argentina striker, who played for Paris Saint-Germain last season, captained Argentina to the World Cup in Qatar last December, ending a 36-year wait for the South American country. Messi, 36, has won the award three times more than anyone else.


Lionel Messi was named the Ballon d’Or 2023 winner at a ceremony in Paris on Monday, the eighth time he has won the prestigious award.
Messi captained Argentina to 2022 World Cup success in Qatar last winter, the first time he won the tournament which ended a 36-year wait for La Albiceleste.
The 36-year-old scored twice in a blockbuster final against France in December which finished 3-3 after extra-time, scored in the penalty shoot-out and was named player of the match.


Messi fought off competition from Manchester City’s Erling Haaland and former Paris Saint-Germain team-mate Kylian Mbappe, as well as 26 other nominees to win the award.
In a league title-winning season for PSG, he played 41 matches during the 2022/23 season, scored 21 goals and provided 20 assists.
At the World Cup in Qatar, Messi scored twice in the group stage and was on target in each knockout round, scoring against Australia, the Netherlands and Croatia, and then in the final.
He finished second in the top goalscorer charts with seven goals and received the Golden Ball at the conclusion of the tournament in recognition of his fine campaign.
On stage, he said: "Thank you very much, to share this with my national team-mates for what we were able to achieve with the national team.
"The entire group, the coaching staff, everyone involved. I’m delighted to be here once more to be able to enjoy this moment one more time.
"To be able to win the World Cup and really achieve my dream, to share this with all those who were involved."
Fittingly, after his World Cup success, he added: "To finish, I want to say happy birthday to Diego Maradona."
After his contract at PSG ran out in the summer, Messi moved to Inter Miami in MLS on a free transfer and was one of three players nominated for the award who do not currently play in Europe.
The former Barcelona forward spent 17 years with the Spanish club between 2004 and 2021 and also won the award in 2009, 2010, 2011, 2012, 2015, 2019 and 2021.
He was 22 years old when he won his first in 2009 and 14 years later has sealed a record eighth award.
Portugal and Al Nassr captain Cristiano Ronaldo is in second place with five awards but was not nominated for the 2023 edition, the first time he has not been up for the award since 2003.
There were plenty of nominations from the Premier League, with Haaland, Kevin De Bruyne and Julian Alvarez among the seven Manchester City players put forward.
"""

# Tokenize text into sentences
sentences = sent_tokenize(text)

# Initialize summary and missed words
summary = ""
missed_words = set()

# Initialize summary length and word frequency
summary_length = 5  # Adjust the length limit as needed
word_frequency = Counter()

# Create a set of stopwords for English
stop_words = set(stopwords.words('english'))

# Sort sentences by density (higher word frequency is denser)
sorted_sentences = sorted(sentences, key=lambda sentence: sum(word_frequency[word.lower()] for word in nltk.word_tokenize(sentence)))

# Create the summary
for sentence in sorted_sentences:
    if len(summary) < summary_length:
        # Calculate the density of the current sentence
        words = nltk.word_tokenize(sentence)
        words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
        density = sum(word_frequency[word] for word in words)

        # Add the sentence to the summary if it doesn't exceed the length limit
        if len(summary) + 1 <= summary_length:
            summary += sentence + " "
            word_frequency.update(words)
        else:
            break

        # Update missed words
        missed_words.update(word for word in words if word_frequency[word] == 0)

        # Print the current summary and missed words
        print("Current Summary:")
        print(summary)
        print("Missed Words:")
        print(missed_words)
    else:
        break

# Display the final summary
print("Final Summary:")
print(summary)

# import nltk
# from nltk.tokenize import sent_tokenize
# from nltk.corpus import stopwords
# from collections import Counter
#
# nltk.download('punkt')
# nltk.download('stopwords')
#
# # Sample text data
# text = """
# Chain of Density (CoD):
# What is CoD? CoD is a technique used in text summarization to create a sequence of summaries with increasing information density while keeping the summary length constant. Each step in the chain adds more relevant details to the summary without exceeding the predetermined length limit.
#
# Purpose and Benefits of CoD:
# Gradual Densification: CoD allows for the gradual densification of summaries. It starts with relatively sparse summaries and incrementally adds more information without making the summary longer.
# Balancing Informativeness and Readability: CoD helps strike a balance between providing informative summaries and maintaining readability. It helps identify the point at which additional information makes a summary too dense or hard to comprehend.
# Informed Decision-Making: By using CoD, businesses and AI-driven solutions like Trevise can obtain feedback from users in a structured and nuanced manner. They can assess how much information users prefer in their feedback and adjust their analysis and response strategies accordingly.
# Improved Customer Service: For a tool like Trevise, which focuses on customer sentiment analysis, CoD can be valuable in fine-tuning its analysis. By understanding at what level of density users prefer their feedback to be summarized, Trevise can better inform businesses about customer sentiments.
# """
#
# # Tokenize text into sentences
# sentences = sent_tokenize(text)
#
# # Initialize summary
# summary = ""
# summary_length = 5  # Adjust the length limit as needed
#
# # Create a set of stopwords for English
# stop_words = set(stopwords.words('english'))
#
# # Calculate word frequency for sentence density
# word_frequency = Counter()
#
# for sentence in sentences:
#     words = nltk.word_tokenize(sentence)
#     words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
#     word_frequency.update(words)
#
# # Sort sentences by density (higher word frequency is denser)
# sentences = sorted(sentences, key=lambda sentence: sum(word_frequency[word.lower()] for word in nltk.word_tokenize(sentence)))
#
# # Create the incremental summary
# incremental_summary = ""
# for sentence in sentences[:summary_length]:
#     incremental_summary += sentence + " "
#     print("Incremental Summary:")
#     print(incremental_summary)
#     print("\n---\n")
#
# # Final summary
# print("Final Summary:")
# print(incremental_summary)


# import nltk
# from nltk.tokenize import sent_tokenize
# from nltk.corpus import stopwords
# from collections import Counter
#
# nltk.download('punkt')
# nltk.download('stopwords')
#
# # Sample text data
# text = """
# The winner of the 2023 Ballon d’Or has been announced and Lionel Messi has won a record-extending eighth award. The Inter Miami and Argentina striker, who played for Paris Saint-Germain last season, captained Argentina to the World Cup in Qatar last December, ending a 36-year wait for the South American country. Messi, 36, has won the award three times more than anyone else.
#
#
# Lionel Messi was named the Ballon d’Or 2023 winner at a ceremony in Paris on Monday, the eighth time he has won the prestigious award.
# Messi captained Argentina to 2022 World Cup success in Qatar last winter, the first time he won the tournament which ended a 36-year wait for La Albiceleste.
# The 36-year-old scored twice in a blockbuster final against France in December which finished 3-3 after extra-time, scored in the penalty shoot-out and was named player of the match.
#
#
# Messi fought off competition from Manchester City’s Erling Haaland and former Paris Saint-Germain team-mate Kylian Mbappe, as well as 26 other nominees to win the award.
# In a league title-winning season for PSG, he played 41 matches during the 2022/23 season, scored 21 goals and provided 20 assists.
# At the World Cup in Qatar, Messi scored twice in the group stage and was on target in each knockout round, scoring against Australia, the Netherlands and Croatia, and then in the final.
# He finished second in the top goalscorer charts with seven goals and received the Golden Ball at the conclusion of the tournament in recognition of his fine campaign.
# On stage, he said: "Thank you very much, to share this with my national team-mates for what we were able to achieve with the national team.
# "The entire group, the coaching staff, everyone involved. I’m delighted to be here once more to be able to enjoy this moment one more time.
# "To be able to win the World Cup and really achieve my dream, to share this with all those who were involved."
# Fittingly, after his World Cup success, he added: "To finish, I want to say happy birthday to Diego Maradona."
# After his contract at PSG ran out in the summer, Messi moved to Inter Miami in MLS on a free transfer and was one of three players nominated for the award who do not currently play in Europe.
# The former Barcelona forward spent 17 years with the Spanish club between 2004 and 2021 and also won the award in 2009, 2010, 2011, 2012, 2015, 2019 and 2021.
# He was 22 years old when he won his first in 2009 and 14 years later has sealed a record eighth award.
# Portugal and Al Nassr captain Cristiano Ronaldo is in second place with five awards but was not nominated for the 2023 edition, the first time he has not been up for the award since 2003.
# There were plenty of nominations from the Premier League, with Haaland, Kevin De Bruyne and Julian Alvarez among the seven Manchester City players put forward.
# """
#
# # Tokenize text into sentences
# sentences = sent_tokenize(text)
#
# # Initialize summary
# summary = ""
# summary_length = 10  # Adjust the length limit as needed
#
# # Create a set of stopwords for English
# stop_words = set(stopwords.words('english'))
#
# # Calculate word frequency for sentence density
# word_frequency = Counter()
#
# for sentence in sentences:
#     words = nltk.word_tokenize(sentence)
#     words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
#     word_frequency.update(words)
#
# # Sort sentences by density (higher word frequency is denser)
# sentences = sorted(sentences, key=lambda sentence: sum(word_frequency[word.lower()] for word in nltk.word_tokenize(sentence)))
#
# # Divide the sentences into equal parts for incremental summaries
# num_parts = len(sentences) // summary_length
# incremental_summaries = []
# for i in range(summary_length):
#     start_idx = i * num_parts
#     end_idx = (i + 1) * num_parts
#     incremental_summary = " ".join(sentences[start_idx:end_idx])
#     incremental_summaries.append(incremental_summary)
#
# # Print incremental summaries
# for i, summary in enumerate(incremental_summaries):
#     print(f"Incremental Summary {i + 1}:")
#     print(summary)
#     print("\n---\n")

