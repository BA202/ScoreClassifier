from DataHandler.DataHandler import DataHandler
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn import svm
import nltk
import numpy as np


import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context


nltk.download('wordnet')
nltk.download('omw-1.4')

testbenchDataHabler = DataHandler("/Users/tobiasrothlin/Documents/BachelorArbeit/DataSets/ClassifiedDataSetV1.3")

katData = testbenchDataHabler.getCategorieData("Location")
stemmer = SnowballStemmer("english")
trainingData = katData[:-100]
verifcationData = katData[-100:]

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

for doc in trainingData:
    processed_docs.append(preprocess(doc[0]))

dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n= 100000)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

lda_model =  gensim.models.LdaModel(bow_corpus,
                                   num_topics = 8,
                                   id2word = dictionary,
                                   passes = 10)


dataView = {lable:[] for lable in {lbl[1] for lbl in trainingData}}
for data in trainingData:
    # Data preprocessing step for the unseen document
    bow_vector = dictionary.doc2bow(preprocess(data[0]))
    results = lda_model[bow_vector]
    dataView[data[1]].append([result[1] for result in results])

print(dataView)

model = svm.SVC(kernel="rbf",C=2.1,gamma=1.5)
vecSen = []
vecLbl = []
for key in dataView.keys():
    for vec in dataView[key]:
        value = [0,0,0,0,0,0,0,0]
        for i,number in enumerate(vec):
            value[i] = number
        vecSen.append(value)
        vecLbl.append(key)

print(np.array(vecSen))
model.fit(np.array(vecSen),vecLbl)

for data in verifcationData:
    bow_vector = dictionary.doc2bow(preprocess(data[0]))
    results = lda_model[bow_vector]
    res = model.predict(np.array([[result[1] for result in results]]))
    print(data[1],res)
