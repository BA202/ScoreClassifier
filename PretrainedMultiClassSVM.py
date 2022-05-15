from transformers import TFBertModel
from transformers import AutoTokenizer
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os
import tensorflow as tf

class PretrainedMultiClassSVM:

    def __init__(self, trainingData=None, **kwargs):


        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

        modelName = 'bert-base-uncased'

        setOfCats = list({sample[1] for sample in trainingData})
        self.__catToInt = {cat: i for i, cat in enumerate(list(setOfCats))}
        self.__intToCat = {self.__catToInt[key]: key for key in self.__catToInt.keys()}

        self.__tokenizer = AutoTokenizer.from_pretrained(modelName)
        self.__preModel = TFBertModel.from_pretrained(modelName,from_pt=True)
        inputVectors = [self.__bertTransformation(sample[0]) for sample in trainingData]
        outputLable = [sample[1] for sample in trainingData]
        print(inputVectors[0])
        #SVM
        print("GridSearchStarted")
        C_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        gamma_range = [1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
        degree = [3]
        gridSearchParmeters = dict(gamma=gamma_range, C=C_range,kernel=['rbf'], degree=degree,class_weight=['balanced'])
        grid_search = GridSearchCV(svm.SVC(),gridSearchParmeters,cv=10, return_train_score=True,n_jobs=-1)
        grid_search.fit(inputVectors, outputLable)

        print("best param are {}".format(grid_search.best_params_))
        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']
        for mean, std, param in zip(means, stds,grid_search.cv_results_['params']):
            print("{} (+/-) {} for {}".format(round(mean, 3),round(std, 2), param))

        self.__model = svm.SVC(gamma=grid_search.best_params_['gamma'],
                               C=grid_search.best_params_['C'],
                               kernel=grid_search.best_params_['kernel'],
                               degree=grid_search.best_params_['degree'],
                               class_weight='balanced')
        self.__model.fit(inputVectors, outputLable)



    def classify(self, sentence):
        prediction = self.__model.predict([self.__bertTransformation(sentence)])
        return prediction[0]

    def __bertTransformation(self,sen):
        inputs = self.__tokenizer(sen, return_tensors="tf")
        output = self.__preModel(inputs)
        return output.last_hidden_state[:, 0][0]

    def getParameters(self):
        modelParams = self.__model.get_params()
        return {'kernel': modelParams['kernel'],
                'degree': modelParams['degree'], 'gamma': modelParams['gamma'],
                'C': modelParams['C'], 'max_iter': modelParams['max_iter']}

