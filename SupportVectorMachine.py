from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from nltk.stem.snowball import GermanStemmer
from nltk.stem import PorterStemmer
from DataHandler.DataHandler import DataHandler


class SupportVectorMachine:

    def __init__(self,lan="English",path="", debug=False,**kwargs):
        localDataHandler = DataHandler(folderPath=path,lan=lan)

        self.__gamma = 0.5
        self.__kernel = 'linear'
        self.__C = 100
        self.__degree = 3
        self.__ListOfKat = ["Location","Room","Food","Staff","ReasonForStay", "GeneralUtility","HotelOrganisation"]
        self.__classifiers = []

        if lan == "German":
            stopwords = ['aber', 'alle', 'allem', 'allen', 'aller', 'alles',
                         'als',
                         'also', 'am', 'an', 'ander', 'andere', 'anderem',
                         'anderen', 'anderer', 'anderes', 'anderm', 'andern',
                         'anderr', 'anders', 'auch', 'auf', 'aus', 'bei',
                         'bin',
                         'bis', 'bist', 'da', 'damit', 'dann', 'der', 'den',
                         'des',
                         'dem', 'die', 'das', 'dass', 'daß', 'derselbe',
                         'derselben', 'denselben', 'desselben', 'demselben',
                         'dieselbe', 'dieselben', 'dasselbe', 'dazu', 'dein',
                         'deine', 'deinem', 'deinen', 'deiner', 'deines',
                         'denn',
                         'derer', 'dessen', 'dich', 'dir', 'du', 'dies',
                         'diese',
                         'diesem', 'diesen', 'dieser', 'dieses', 'doch',
                         'dort',
                         'durch', 'ein', 'eine', 'einem', 'einen', 'einer',
                         'eines', 'einig', 'einige', 'einigem', 'einigen',
                         'einiger', 'einiges', 'einmal', 'er', 'ihn', 'ihm',
                         'es',
                         'etwas', 'euer', 'eure', 'eurem', 'euren', 'eurer',
                         'eures', 'für', 'gegen', 'gewesen', 'hab', 'habe',
                         'haben', 'hat', 'hatte', 'hatten', 'hier', 'hin',
                         'hinter', 'ich', 'mich', 'mir', 'ihr', 'ihre',
                         'ihrem',
                         'ihren', 'ihrer', 'ihres', 'euch', 'im', 'in',
                         'indem',
                         'ins', 'ist', 'jede', 'jedem', 'jeden', 'jeder',
                         'jedes',
                         'jene', 'jenem', 'jenen', 'jener', 'jenes', 'jetzt',
                         'kann', 'kein', 'keine', 'keinem', 'keinen', 'keiner',
                         'keines', 'können', 'könnte', 'machen', 'man',
                         'manche',
                         'manchem', 'manchen', 'mancher', 'manches', 'mein',
                         'meine', 'meinem', 'meinen', 'meiner', 'meines',
                         'mit',
                         'muss', 'musste', 'nach', 'nicht', 'nichts', 'noch',
                         'nun', 'nur', 'ob', 'oder', 'ohne', 'sehr', 'sein',
                         'seine', 'seinem', 'seinen', 'seiner', 'seines',
                         'selbst',
                         'sich', 'sie', 'ihnen', 'sind', 'so', 'solche',
                         'solchem',
                         'solchen', 'solcher', 'solches', 'soll', 'sollte',
                         'sondern', 'sonst', 'über', 'um', 'und', 'uns',
                         'unsere',
                         'unserem', 'unseren', 'unser', 'unseres', 'unter',
                         'viel',
                         'vom', 'von', 'vor', 'während', 'war', 'waren',
                         'warst',
                         'was', 'weg', 'weil', 'weiter', 'welche', 'welchem',
                         'welchen', 'welcher', 'welches', 'wenn', 'werde',
                         'werden', 'wie', 'wieder', 'will', 'wir', 'wird',
                         'wirst',
                         'wo', 'wollen', 'wollte', 'würde', 'würden', 'zu',
                         'zum',
                         'zur', 'zwar', 'zwischen']
            self.__stemmer = GermanStemmer()
        else:
            stopwords = ['from', 'him', 'shouldn', 'will', 'than', 'what',
                         'weren',
                         'over', "shouldn't", 'in', 'its', 'above', 'o',
                         'about',
                         'has', 'or', 'off', 'before', 'doesn', "you've",
                         'just',
                         'but', 'my', 'd', 'having', 'they', "aren't",
                         'ourselves',
                         'ain', 'with', "haven't", 'it', 'under', 'after',
                         'myself', 'did', "you'd", "won't", 'which', 'theirs',
                         've', 'ours', 'haven', "mustn't", 'same', "it's",
                         'mightn', 'for', 'our', 'how', "don't", 'both',
                         'them',
                         'those', 'ma', 'she', 'any', 'once', 'couldn',
                         'these',
                         'itself', 'then', "doesn't", 'i', 'here', 'shan',
                         'so',
                         'hadn', 'who', 'into', "she's", 'yourselves', 'me',
                         'all',
                         'mustn', 'own', 'isn', 'needn', 'y', 'be', "weren't",
                         'against', 'up', 'whom', 'where', 'wouldn', 'by',
                         "hasn't", "you'll", 'because', 'at', 'such', "you're",
                         'their', 'down', 'doing', 'through', 'not', "shan't",
                         'is', "couldn't", 'most', 'to', "needn't", 'again',
                         's',
                         'was', 'until', 'no', 'themselves', 'and', 'few',
                         "mightn't", 'does', 'out', 'too', "that'll",
                         'between',
                         're', 'further', 'why', 'aren', 'his', 'll',
                         'himself',
                         'while', 'should', "should've", 'there', 'nor', 'he',
                         'yourself', 'other', "wasn't", 'more', 'her', 'very',
                         'have', 'during', 'only', 'hasn', 'when', 'an',
                         'didn',
                         'below', 'on', 'being', "isn't", 'had', 'that', 'do',
                         'of', 'the', 'were', 'now', "wouldn't", 'your',
                         'some',
                         't', 'we', 'won', 'yours', "didn't", 'been', 'm',
                         'as',
                         'wasn', 'if', "hadn't", 'am', 'are', 'this', 'you',
                         'hers', 'each', 'don', 'can', 'a', 'herself']
            self.__stemmer = PorterStemmer()


        self.__vectorizer = TfidfVectorizer(max_features=2500, min_df=5,max_df=0.8, sublinear_tf=True,use_idf=True,stop_words=stopwords)
        train_vectors = self.__vectorizer.fit_transform([sample[0] for sample in localDataHandler.getScoreData()])

        for kat in self.__ListOfKat:
            self.__classifiers.append(svm.SVC(gamma=self.__gamma, C=self.__C,kernel=self.__kernel, degree=self.__degree,class_weight='balanced'))
            labels = [sample[1] for sample in localDataHandler.getCategorieData(kat)]
            temp = []
            for label in labels:
                if label == kat:
                    temp.append("True")
                else:
                    temp.append("Not")

            self.__classifiers[-1].fit(train_vectors,temp)

    def classify(self, sentence):
        senVector = self.__vectorizer.transform([self.cleanUp(sentence)])
        res = {}

        for i,classifier in enumerate(self.__classifiers):
            if classifier.predict(senVector)[0] == "True":
                res[self.__ListOfKat[i]] = True
        print(res)
        if len(list(res.keys())) == 0:
            return "Unknown", 1
        else:
            return list(res.keys())[0], 1/len(list(res.keys()))


    def cleanUp(self, sen):
        sen = sen.lower()
        lem = self.__stemmer.stem(sen)
        return lem

    def getParameters(self):
        modelParams = self.__classifiers[0].get_params()
        return {'kernel': modelParams['kernel'],
                'degree': modelParams['degree'], 'gamma': modelParams['gamma'],
                'C': modelParams['C'], 'max_iter': modelParams['max_iter']}


if __name__ == '__main__':
    myClassifier = SupportVectorMachine("English","/Users/tobiasrothlin/Documents/BachelorArbeit/DataSets/ClassifiedDataSetV1.3")
    print(myClassifier.classify("the train station was near"))
