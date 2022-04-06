from DataHandler.DataHandler import DataHandler
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn import svm
import nltk
import numpy as np
import os, ssl

class LDAToken:

    def __init__(self,trainingData= None,**kwargs):

        if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
                getattr(ssl, '_create_unverified_context', None)):
            ssl._create_default_https_context = ssl._create_unverified_context

        nltk.download('wordnet')
        nltk.download('omw-1.4')

        processed_docs = []

        for doc in trainingData:
            processed_docs.append(self.preprocess(doc[0]))

        self.__dictionary = gensim.corpora.Dictionary(processed_docs)
        self.__dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n=100000)

        bow_corpus = [self.__dictionary.doc2bow(doc) for doc in processed_docs]

        self.__lda_model = gensim.models.LdaModel(bow_corpus,
                                           num_topics=8,
                                           id2word=self.__dictionary,
                                           passes=10)

        dataView = {lable: [] for lable in {lbl[1] for lbl in trainingData}}
        for data in trainingData:
            # Data preprocessing step for the unseen document
            bow_vector = self.__dictionary.doc2bow(LDAToken.preprocess(data[0]))
            results = self.__lda_model[bow_vector]
            dataView[data[1]].append([result[1] for result in results])

        self.__model = svm.SVC(kernel="linear", C=2.1, gamma=1.5)
        vecSen = []
        vecLbl = []
        for key in dataView.keys():
            for vec in dataView[key]:
                value = [0, 0, 0, 0, 0, 0, 0, 0]
                for i, number in enumerate(vec):
                    value[i] = number
                vecSen.append(value)
                vecLbl.append(key)

        self.__model.fit(np.array(vecSen), vecLbl)


    def classify(self,sentence):
        bow_vector = self.__dictionary.doc2bow(LDAToken.preprocess(sentence))
        results = self.__lda_model[bow_vector]
        value = [0, 0, 0, 0, 0, 0, 0, 0]
        for i, number in enumerate([result[1] for result in results]):
            value[i] = number

        res = self.__model.predict(np.array([value]))
        return res[0]


    def getParameters(self):
        return {'kernel':'linear'}

    @staticmethod
    def lemmatize_stemming(text):
        stemmer = SnowballStemmer("english")
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

    # Tokenize and lemmatize
    @staticmethod
    def preprocess(text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(
                    token) > 3:
                result.append(LDAToken.lemmatize_stemming(token))

        return result
