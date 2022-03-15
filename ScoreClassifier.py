import nltk
import sklearn
import ssl
from sklearn.naive_bayes import MultinomialNB


class ScoreClassifier:

    def __init__(self,trainingData= None):
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        nltk.download('stopwords')

        self.__lutClassToId = {
            'Positiv': 1,
            'Neutral':0,
            'Negative':-1
        }
        self.__lutIdToClass = {self.__lutClassToId[key]:key for  key in self.__lutClassToId.keys()}

        self.__trainingData = trainingData

        if not self.__trainingData == None:
            self.__steamedTrainingData = []

            for data in self.__trainingData:
                self.__steamedTrainingData.append([self.cleanUp(data[0]),data[1]])

            self.__vectorizer = sklearn.feature_extraction.text.CountVectorizer(binary=False, ngram_range=(1, 1))
            self.__tf_features_train = self.__vectorizer.fit_transform([sen[0] for sen in self.__steamedTrainingData])

            self.__model = MultinomialNB()
            self.__model.fit(self.__tf_features_train, [sen[1] for sen in self.__steamedTrainingData])


    def classify(self,sentence):
        clean = self.cleanUp(sentence)
        vec = self.featureExctractor(clean)
        return self.__model.predict(vec)[0]

    def classToId(self,str):
        if str in self.__lutClassToId.keys():
            return self.__lutClassToId[str]
        else:
            raise TypeError(f"Invalid Classification: {str}")

    def idToClass(self,id):
        if id in self.__lutIdToClass.keys():
            return self.__lutIdToClass[id]
        else:
            raise TypeError(f"Invalid Id: {id}")

    def cleanUp(self,sen):
        sen = sen.lower()
        english_stop_words = nltk.corpus.stopwords.words('english')
        for stopWord in english_stop_words:
            stopWord = ' ' + stopWord + ' '
            sen = sen.replace(stopWord, ' ')
        stemmer = nltk.porter.PorterStemmer()
        stemmed = ' '.join([stemmer.stem(token) for token in sen.split()])
        return stemmed

    def featureExctractor(self,sen):
        tf_features_test = self.__vectorizer.transform([sen])
        return tf_features_test



