from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import gensim
import numpy as np
from sklearn import svm
from nltk.stem import PorterStemmer
from sklearn.model_selection import GridSearchCV
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer
from tensorflow.keras.utils import to_categorical
from scipy.sparse import csr_matrix, vstack
from WordEmbedding import WordEmbedding




class SupportVectorMachineVectoriserTest:

    def __init__(self,trainingData= None, kernel = 'linear',debug = False,**kwargs):
        stopwords = ['from', 'him', 'shouldn', 'will', 'than', 'what', 'weren',
                     'over', "shouldn't", 'in', 'its', 'above', 'o', 'about',
                     'has', 'or', 'off', 'before', 'doesn', "you've", 'just',
                     'but', 'my', 'd', 'having', 'they', "aren't", 'ourselves',
                     'ain', 'with', "haven't", 'it', 'under', 'after',
                     'myself', 'did', "you'd", "won't", 'which', 'theirs',
                     've', 'ours', 'haven', "mustn't", 'same', "it's",
                     'mightn', 'for', 'our', 'how', "don't", 'both', 'them',
                     'those', 'ma', 'she', 'any', 'once', 'couldn', 'these',
                     'itself', 'then', "doesn't", 'i', 'here', 'shan', 'so',
                     'hadn', 'who', 'into', "she's", 'yourselves', 'me', 'all',
                     'mustn', 'own', 'isn', 'needn', 'y', 'be', "weren't",
                     'against', 'up', 'whom', 'where', 'wouldn', 'by',
                     "hasn't", "you'll", 'because', 'at', 'such', "you're",
                     'their', 'down', 'doing', 'through', 'not', "shan't",
                     'is', "couldn't", 'most', 'to', "needn't", 'again', 's',
                     'was', 'until', 'no', 'themselves', 'and', 'few',
                     "mightn't", 'does', 'out', 'too', "that'll", 'between',
                     're', 'further', 'why', 'aren', 'his', 'll', 'himself',
                     'while', 'should', "should've", 'there', 'nor', 'he',
                     'yourself', 'other', "wasn't", 'more', 'her', 'very',
                     'have', 'during', 'only', 'hasn', 'when', 'an', 'didn',
                     'below', 'on', 'being', "isn't", 'had', 'that', 'do',
                     'of', 'the', 'were', 'now', "wouldn't", 'your', 'some',
                     't', 'we', 'won', 'yours', "didn't", 'been', 'm', 'as',
                     'wasn', 'if', "hadn't", 'am', 'are', 'this', 'you',
                     'hers', 'each', 'don', 'can', 'a', 'herself']

        self.__stemmer = PorterStemmer()
        modelName = 'bert-base-uncased'

        self.__tokenizer = WordEmbedding("WordEmbeddings/CBOWEmbedding")
        self.__trainingData = [self.__tokenizer.vectorize(sample[0])for sample in trainingData]
        #self.__tokenizer = AutoTokenizer.from_pretrained(modelName)
        #self.__trainingData = self.__tokenizer([sample[0] for sample in trainingData])
        #oneHotOutput = np.array([sum(to_categorical(sample,num_classes=len(self.__tokenizer.get_vocab()))) for sample in self.__trainingData['input_ids']])
        #self.__trainingData = oneHotOutput
        C_range = np.linspace(0.1,1,10)
        gamma_range = np.linspace(0.1,10,10)
        degree = [3]
        gridSearchParmeters = dict(gamma=gamma_range, C=C_range,
                                   kernel=[kernel], degree=degree,
                                   class_weight=['balanced'])

        grid_search = GridSearchCV(svm.SVC(),
                                   gridSearchParmeters,
                                   cv=10, return_train_score=True,
                                   n_jobs=-1)
        grid_search.fit(self.__trainingData, [sample[1] for sample in
                                        trainingData])

        print("best param are {}".format(grid_search.best_params_))
        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']
        for mean, std, param in zip(means, stds,
                                    grid_search.cv_results_['params']):
            print("{} (+/-) {} for {}".format(round(mean, 3),
                                              round(std, 2), param))

        self.__model = svm.SVC(gamma=grid_search.best_params_['gamma'],
                               C=grid_search.best_params_['C'],
                               kernel=grid_search.best_params_['kernel'],
                               degree=grid_search.best_params_['degree'],
                               class_weight='balanced')
        self.__model.fit(self.__trainingData, [sample[1] for sample in
                                         trainingData])



    def classify(self,sentence):
        #vec = self.__tokenizer([sentence])
        vec = self.__tokenizer.vectorize(sentence)

        #oneHotOutput = np.array([sum(to_categorical(sample, num_classes=len(self.__tokenizer.get_vocab()))) for sample in vec['input_ids']])
        #vec = oneHotOutput
        prediction = self.__model.predict([vec])
        return prediction[0]



    def cleanUp(self,sen):
        sen = sen.lower()
        lem = self.__stemmer.stem(sen)
        return lem

    @staticmethod
    def lossFunction(y_true, y_pred):
        error = 0
        for yT,yP in zip(y_true,y_pred):
            factor = 1
            if yT == "Negative":
                factor = 100

            if not yT == yP:
                error += factor
        return error

    def getParameters(self):
        modelParams = self.__model.get_params()
        return {'kernel':modelParams['kernel'],'degree':modelParams['degree'],'gamma':modelParams['gamma'],'C':modelParams['C'],'max_iter':modelParams['max_iter']}
