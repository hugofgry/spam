import pandas as pd
import string
from nltk.corpus import stopwords
from collections import Counter


# Remove punctuation and stopwords
def remove_punctuation_and_stopwords(sms):
    sms_no_punctuation = [i for i in sms if i not in string.punctuation]
    sms_no_punctuation = "".join(sms_no_punctuation).split()
    sms_no_punctuation_no_stopwords = [word.lower() for word in sms_no_punctuation if word.lower() not in stopwords.words("english")]
    return sms_no_punctuation_no_stopwords


# Count the ham words without punctuation and stopwords
def counter_ham_words(data):
    list_ham_words = []
    for sublist in data:
        for item in sublist:
            list_ham_words.append(item)
    count_ham  = Counter(list_ham_words)
    df_hamwords_top20  = pd.DataFrame(count_ham.most_common(20),  columns=['word', 'count'])
    return df_hamwords_top20


# Count the spam words without punctuation and stopwords
def counter_spam_words(data):
    list_spam_words = []
    for sublist in data:
        for item in sublist:
            list_spam_words.append(item)
    count_spam  = Counter(list_spam_words)
    df_spamwords_top20  = pd.DataFrame(count_spam.most_common(20),  columns=['word', 'count'])
    return df_spamwords_top20