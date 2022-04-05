from DataHandler.DataHandler import DataHandler
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import nltk


import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context


nltk.download('wordnet')
nltk.download('omw-1.4')

testbenchDataHabler = DataHandler("/Users/tobiasrothlin/Documents/BachelorArbeit/DataSets/ClassifiedDataSetV1.3")

katData = testbenchDataHabler.getCategorieData("Location")
stemmer = SnowballStemmer("english")
testData = katData[-100:]
katData = katData[:-100]

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


# Tokenize and lemmatize
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(
                token) > 3:
            result.append(lemmatize_stemming(token))

    return result




processed_docs = []

for doc in katData:
    processed_docs.append(preprocess(doc[0]))

dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n= 100000)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

lda_model =  gensim.models.LdaModel(bow_corpus,
                                   num_topics = 8,
                                   id2word = dictionary,
                                   passes = 10)



for data in testData:
    # Data preprocessing step for the unseen document
    bow_vector = dictionary.doc2bow(preprocess(data[0]))
    print(f"--{data[1]:20}-->",end="")
    for index, score in [sorted(lda_model[bow_vector],key=lambda tup: -1 * tup[1])[0]]:
        print("Topic: {}".format(lda_model.print_topic(index, 5)))
