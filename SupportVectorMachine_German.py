from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from nltk.stem.snowball import GermanStemmer
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import fbeta_score, make_scorer


class SupportVectorMachine:

    def __init__(self,trainingData= None, kernel = 'linear',debug = False,**kwargs):
        stopwords = ['aber', 'alle', 'allem', 'allen', 'aller', 'alles', 'als',
                     'also', 'am', 'an', 'ander', 'andere', 'anderem',
                     'anderen','anderer', 'anderes', 'anderm', 'andern',
                     'anderr', 'anders', 'auch', 'auf', 'aus', 'bei', 'bin',
                     'bis', 'bist', 'da', 'damit', 'dann', 'der', 'den', 'des',
                     'dem', 'die', 'das', 'dass', 'daß', 'derselbe',
                     'derselben', 'denselben', 'desselben', 'demselben',
                     'dieselbe', 'dieselben', 'dasselbe', 'dazu', 'dein',
                     'deine', 'deinem', 'deinen', 'deiner', 'deines', 'denn',
                     'derer', 'dessen', 'dich', 'dir', 'du', 'dies', 'diese',
                     'diesem', 'diesen', 'dieser', 'dieses', 'doch', 'dort',
                     'durch', 'ein', 'eine', 'einem', 'einen', 'einer',
                     'eines', 'einig', 'einige', 'einigem', 'einigen',
                     'einiger', 'einiges', 'einmal', 'er', 'ihn', 'ihm', 'es',
                     'etwas', 'euer', 'eure', 'eurem', 'euren', 'eurer',
                     'eures', 'für', 'gegen', 'gewesen', 'hab', 'habe',
                     'haben', 'hat', 'hatte', 'hatten', 'hier', 'hin',
                     'hinter', 'ich', 'mich', 'mir', 'ihr', 'ihre', 'ihrem',
                     'ihren', 'ihrer', 'ihres', 'euch', 'im', 'in', 'indem',
                     'ins', 'ist', 'jede', 'jedem', 'jeden', 'jeder', 'jedes',
                     'jene', 'jenem', 'jenen', 'jener', 'jenes', 'jetzt',
                     'kann', 'kein', 'keine', 'keinem', 'keinen', 'keiner',
                     'keines', 'können', 'könnte', 'machen', 'man', 'manche',
                     'manchem', 'manchen', 'mancher', 'manches', 'mein',
                     'meine', 'meinem', 'meinen', 'meiner', 'meines', 'mit',
                     'muss', 'musste', 'nach', 'nicht', 'nichts', 'noch',
                     'nun', 'nur', 'ob', 'oder', 'ohne', 'sehr', 'sein',
                     'seine', 'seinem', 'seinen', 'seiner', 'seines', 'selbst',
                     'sich', 'sie', 'ihnen', 'sind', 'so', 'solche', 'solchem',
                     'solchen', 'solcher', 'solches', 'soll', 'sollte',
                     'sondern', 'sonst', 'über', 'um', 'und', 'uns', 'unsere',
                     'unserem', 'unseren', 'unser', 'unseres', 'unter', 'viel',
                     'vom', 'von', 'vor', 'während', 'war', 'waren', 'warst',
                     'was', 'weg', 'weil', 'weiter', 'welche', 'welchem',
                     'welchen', 'welcher', 'welches', 'wenn', 'werde',
                     'werden', 'wie', 'wieder', 'will', 'wir', 'wird', 'wirst',
                     'wo', 'wollen', 'wollte', 'würde', 'würden', 'zu', 'zum',
                     'zur', 'zwar', 'zwischen']

        self.__trainingData = trainingData
        self.__stemmer = GermanStemmer()
        self.__doGridSearch = False

        if 'param' in kwargs.keys():
            kernel = kwargs['param']

        if kernel == "rbf":
            self.__doGridSearch = True

        elif kernel == "poly":
            self.__doGridSearch = True

        elif kernel == "sigmoid":
            self.__doGridSearch = True


        if not self.__trainingData == None:
            self.__vectorizer = TfidfVectorizer(max_features=2500,min_df=5,max_df=0.8,sublinear_tf=True,use_idf=True,stop_words=stopwords)

            train_vectors = self.__vectorizer.fit_transform([sample[0] for sample in self.__trainingData])
            if self.__doGridSearch:
                score = make_scorer(self.lossFunction,
                                    greater_is_better=False)
                if kernel == "rbf" or kernel == "poly" or kernel == "sigmoid":
                    C_range = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
                    gamma_range = [1,1.2,1.3,1.4,1.5,1.6,1.7]
                    degree = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                    gridSearchParmeters = dict(gamma=gamma_range, C=C_range,kernel = [kernel],degree= degree,class_weight=['balanced'])
                else:
                    gridSearchParmeters = {}
                    raise ValueError("Not able to perform grid search!")

                grid_search = GridSearchCV(svm.SVC(),
                                           gridSearchParmeters,
                                           cv=10, return_train_score=True,
                                           n_jobs=-1)
                grid_search.fit(train_vectors, [sample[1] for sample in
                                                self.__trainingData])

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
                                       degree=grid_search.best_params_['degree'],class_weight='balanced')
                self.__model.fit(train_vectors, [sample[1] for sample in
                                                 self.__trainingData])

            else:
                self.__model = svm.SVC(kernel=kernel)
                if debug:
                    print(f"---SupportVectorMachine with kernel: {kernel}")
                self.__model.fit(train_vectors, [sample[1] for sample in self.__trainingData])


    def classify(self,sentence):
        senVector = self.__vectorizer.transform([self.cleanUp(sentence)])
        prediction = self.__model.predict(senVector)
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
