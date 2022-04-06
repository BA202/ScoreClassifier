from MultinomialNaiveBayes import MultinomialNaiveBayes
from Vader import Vader
from SupportVectorMachine import SupportVectorMachine
from RandomForest import RandomForest
from RNN import RNN
from ZeroShotClassification import ZeroShotClassification
from TransformesModel import TransformesModel
from DataHandler.DataHandler import DataHandler
from ModelReport.ModelReport import ModelReport
from BERT_Transformers import BERT_Transformers
from LDAToken import LDAToken
import random
import traceback
import json

class testConstants:
    folds = 10
    seed = 4.83819
    dataLocation = "/home/student/Desktop/Data"
    balancedDataSet = False
    balancedSplitDataSet = False

    modelsToEvaluate = [
        {
            'data': 'Score',
            'model': MultinomialNaiveBayes,
            'modelName': "MultinomialNaiveBayesOnScore",
            'modelCreator': "Tobias Rothlin",
            'mlPrinciple': "Multinomial Naive Bayes",
            'refrences': {
                'NultinomialNB Explained': "https://towardsdatascience.com/sentimental-analysis-using-vader-a3415fef7664",
                'Stanford NLP Course': "http://spark-public.s3.amazonaws.com/nlp/slides/naivebayes.pdf",
                'Stanford NLP Lecture': "https://www.youtube.com/playlist?list=PLLssT5z_DsK8HbD2sPcUIDfQ7zmBarMYv",
                'Engilsh Stopwords': "https://www.tutorialspoint.com/python_text_processing/python_remove_stopwords.htm"
            },
            'algorithemDescription': """The learning algorithm used in this classification is the Multinomial Naïve Bayes. This approach was chosen as it is easy to implement and is computational very efficient. The first step in the classification pipeline is removing all strop words for example 'i', 'me', 'my', 'myself', etc. A list of English stop word is provided by the nltk module. The stop words remover just removes every word that is in the list of stop words. Next the sentence is passed through the stemmer. Stemmers remove morphological affixes from words, leaving only the word stem. This is done with the PorterStemmer class from the nltk module. The final preprocessing step is to vectorize the sentence. This results in a bag of words representation of the sentence. First all the words must be tokenized and then counted. The result will be a numerical feature vector. To generate this vector the CountVectorizer class from sklearn is used.  This class implements both tokenization and occurrence counting in a single class. With the sentence now represented in a vector the Naïve Bayes classifier can work with this vector. For the implementation of the Naïve Bayes classifier the MultinomialNB class (sklearn) is used. """,
            'graphicPath': "/Users/tobiasrothlin/Documents/BachelorArbeit/ScoreClassifier/OverviewImg.png",
            'graphicDescription': "Classification Pipeline",
            'dataSet': f"ClassifiedDataSetV1.3 with {folds} folds cross validation",
            'seed': seed,
            'kfolds': folds,
            'opParams': None
        },
        {
            'data': 'Category',
            'model': MultinomialNaiveBayes,
            'modelName': "MultinomialNaiveBayesOnLocation",
            'modelCreator': "Tobias Rothlin",
            'mlPrinciple': "Multinomial Naive Bayes",
            'refrences': {
                'NultinomialNB Explained': "https://towardsdatascience.com/sentimental-analysis-using-vader-a3415fef7664",
                'Stanford NLP Course': "http://spark-public.s3.amazonaws.com/nlp/slides/naivebayes.pdf",
                'Stanford NLP Lecture': "https://www.youtube.com/playlist?list=PLLssT5z_DsK8HbD2sPcUIDfQ7zmBarMYv",
                'Engilsh Stopwords': "https://www.tutorialspoint.com/python_text_processing/python_remove_stopwords.htm"
            },
            'algorithemDescription': """The learning algorithm used in this classification is the Multinomial Naïve Bayes. This approach was chosen as it is easy to implement and is computational very efficient. The first step in the classification pipeline is removing all strop words for example 'i', 'me', 'my', 'myself', etc. A list of English stop word is provided by the nltk module. The stop words remover just removes every word that is in the list of stop words. Next the sentence is passed through the stemmer. Stemmers remove morphological affixes from words, leaving only the word stem. This is done with the PorterStemmer class from the nltk module. The final preprocessing step is to vectorize the sentence. This results in a bag of words representation of the sentence. First all the words must be tokenized and then counted. The result will be a numerical feature vector. To generate this vector the CountVectorizer class from sklearn is used.  This class implements both tokenization and occurrence counting in a single class. With the sentence now represented in a vector the Naïve Bayes classifier can work with this vector. For the implementation of the Naïve Bayes classifier the MultinomialNB class (sklearn) is used. """,
            'graphicPath': "/Users/tobiasrothlin/Documents/BachelorArbeit/ScoreClassifier/OverviewImg.png",
            'graphicDescription': "Classification Pipeline",
            'dataSet': f"ClassifiedDataSetV1.3 with {folds} folds cross validation",
            'seed': seed,
            'kfolds': folds,
            'opParams': None
        },
        {
            'data': 'Score',
            'model': SupportVectorMachine,
            'modelName': "SupportVectorMachineOnScore",
            'modelCreator': "Tobias Rothlin",
            'mlPrinciple': "Support Vector Machine",
            'refrences': {
                'Sentiment Analysis SVM': "https://medium.com/@vasista/sentiment-analysis-using-svm-338d418e3ff1",
                'Scikit SVM Kernels': "https://scikit-learn.org/stable/modules/svm.html#svm-kernels",
                'Scikit feature extraction': "https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction",
                'Scikit Vectorizer': "https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html"
            },
            'algorithemDescription': """Support vector machines are a robust supervised learning model based on statistical learning. The idea is to find a Hyperplane separating the different classes with the most separation between the closest points. Before the SVM can classify a sentence, the sentence needs to be vectorised. To accomplish the Scikit learn, Tfidf Vectorizer is used. The Vectorizer converts the sentence to a fixed feature vector. With the vectorised sentences, the model can be trained. The best hyperplanes are found in the training step based on the training data. The flexibility of the hyperplane can be defined by the Kernel (linear, sigmoid, RBF). RBF is used for non-linear problems and is also a general-purpose kernel. This model uses a {kernel} kernel.""",
            'graphicPath': "/Users/tobiasrothlin/Documents/BachelorArbeit/ScoreClassifier/SVMPipeline.png",
            'graphicDescription': "Classification Pipeline",
            'dataSet': f"ClassifiedDataSetV1.3 with {folds} folds cross validation",
            'seed': seed,
            'kfolds': folds,
            'opParams': ['rbf','poly','linear', 'sigmoid']
        },
        {
            'data': 'Category',
            'model': SupportVectorMachine,
            'modelName': "SupportVectorMachineOnLocation",
            'modelCreator': "Tobias Rothlin",
            'mlPrinciple': "Support Vector Machine",
            'refrences': {
                'Sentiment Analysis SVM': "https://medium.com/@vasista/sentiment-analysis-using-svm-338d418e3ff1",
                'Scikit SVM Kernels': "https://scikit-learn.org/stable/modules/svm.html#svm-kernels",
                'Scikit feature extraction': "https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction",
                'Scikit Vectorizer': "https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html"
            },
            'algorithemDescription': """Support vector machines are a robust supervised learning model based on statistical learning. The idea is to find a Hyperplane separating the different classes with the most separation between the closest points. Before the SVM can classify a sentence, the sentence needs to be vectorised. To accomplish the Scikit learn, Tfidf Vectorizer is used. The Vectorizer converts the sentence to a fixed feature vector. With the vectorised sentences, the model can be trained. The best hyperplanes are found in the training step based on the training data. The flexibility of the hyperplane can be defined by the Kernel (linear, sigmoid, RBF). RBF is used for non-linear problems and is also a general-purpose kernel. This model uses a {kernel} kernel.""",
            'graphicPath': "/Users/tobiasrothlin/Documents/BachelorArbeit/ScoreClassifier/SVMPipeline.png",
            'graphicDescription': "Classification Pipeline",
            'dataSet': f"ClassifiedDataSetV1.3 with {folds} folds cross validation",
            'seed': seed,
            'kfolds': folds,
            'opParams': ['poly', 'rbf', 'linear', 'sigmoid']
        },
        {
            'data': 'Score',
            'model': RandomForest,
            'modelName': "RandomForestOnScore",
            'modelCreator': "Tobias Rothlin",
            'mlPrinciple': "Random Forest",
            'refrences': {
                'Sentiment Analysis Random Forest': "https://medium.com/@dilip.voleti/sentiment-analysis-using-natural-language-processing-f05b19c2a31d"
            },
            'algorithemDescription': """A Random Forest merges a collection of independent decision trees to get a more accurate and stable predictions.""",
            'graphicPath': "/Users/tobiasrothlin/Documents/BachelorArbeit/ScoreClassifier/SVMPipeline.png",
            'graphicDescription': "Classification Pipeline",
            'dataSet': f"ClassifiedDataSetV1.3 with {folds} folds cross validation",
            'seed': seed,
            'kfolds': folds,
            'opParams': None
        },
        {
            'data': 'Category',
            'model': RandomForest,
            'modelName': "RandomForestOnLocation",
            'modelCreator': "Tobias Rothlin",
            'mlPrinciple': "Random Forest",
            'refrences': {
                'Sentiment Analysis Random Forest': "https://medium.com/@dilip.voleti/sentiment-analysis-using-natural-language-processing-f05b19c2a31d"
            },
            'algorithemDescription': """A Random Forest merges a collection of independent decision trees to get a more accurate and stable predictions.""",
            'graphicPath': "/Users/tobiasrothlin/Documents/BachelorArbeit/ScoreClassifier/SVMPipeline.png",
            'graphicDescription': "Classification Pipeline",
            'dataSet': f"ClassifiedDataSetV1.3 with {folds} folds cross validation",
            'seed': seed,
            'kfolds': folds,
            'opParams': None
        },
        {
            'data': 'Score',
            'model': RNN,
            'modelName': "RNNOnScore",
            'modelCreator': "Tobias Rothlin",
            'mlPrinciple': "RNN",
            'refrences': {
                'Text Classification with TF': "https://www.tensorflow.org/text/tutorials/text_classification_rnn"
            },
            'algorithemDescription': """A neural network that is intentionally run multiple times, where parts of each run feed into the next run. Specifically, hidden layers from the previous run provide part of the input to the same hidden layer in the next run. Recurrent neural networks are particularly useful for evaluating sequences, so that the hidden layers can learn from previous runs of the neural network on earlier parts of the sequence.""",
            'graphicPath': "/Users/tobiasrothlin/Documents/BachelorArbeit/ScoreClassifier/SVMPipeline.png",
            'graphicDescription': "Classification Pipeline",
            'dataSet': f"ClassifiedDataSetV1.3 with {folds} folds cross validation",
            'seed': seed,
            'kfolds': folds,
            'opParams': None
        },
        {
            'data': 'Score',
            'model': TransformesModel,
            'modelName': "TransformesModel",
            'modelCreator': "Tobias Rothlin",
            'mlPrinciple': "TransformesModel",
            'refrences': {
                'Text Classification with TF': "https://www.tensorflow.org/text/tutorials/text_classification_rnn"
            },
            'algorithemDescription': """A neural network that is intentionally run multiple times, where parts of each run feed into the next run. Specifically, hidden layers from the previous run provide part of the input to the same hidden layer in the next run. Recurrent neural networks are particularly useful for evaluating sequences, so that the hidden layers can learn from previous runs of the neural network on earlier parts of the sequence.""",
            'graphicPath': "/Users/tobiasrothlin/Documents/BachelorArbeit/ScoreClassifier/TransformerPipeline.png",
            'graphicDescription': "Classification Pipeline",
            'dataSet': f"ClassifiedDataSetV1.3 with {folds} folds cross validation",
            'seed': seed,
            'kfolds': folds,
            'opParams': None
        },
        {
            'data': 'Score',
            'model': BERT_Transformers,
            'modelName': "BERT_Transformers",
            'modelCreator': "Tobias Rothlin",
            'mlPrinciple': "BERT_Transformers",
            'refrences': {
                'Text Classification with TF': "https://www.tensorflow.org/text/tutorials/text_classification_rnn"
            },
            'algorithemDescription': """A neural network that is intentionally run multiple times, where parts of each run feed into the next run. Specifically, hidden layers from the previous run provide part of the input to the same hidden layer in the next run. Recurrent neural networks are particularly useful for evaluating sequences, so that the hidden layers can learn from previous runs of the neural network on earlier parts of the sequence.""",
            'graphicPath': "/Users/tobiasrothlin/Documents/BachelorArbeit/ScoreClassifier/TransformerPipeline.png",
            'graphicDescription': "Classification Pipeline",
            'dataSet': f"ClassifiedDataSetV1.3 with {folds} folds cross validation",
            'seed': seed,
            'kfolds': folds,
            'opParams': None
        },
        {
            'data': 'Category',
            'model': LDAToken,
            'modelName': "LDAToken",
            'modelCreator': "Tobias Rothlin",
            'mlPrinciple': "LDA with SVM",
            'refrences': {
                'Text Classification with TF': "https://www.tensorflow.org/text/tutorials/text_classification_rnn"
            },
            'algorithemDescription': """A neural network that is intentionally run multiple times, where parts of each run feed into the next run. Specifically, hidden layers from the previous run provide part of the input to the same hidden layer in the next run. Recurrent neural networks are particularly useful for evaluating sequences, so that the hidden layers can learn from previous runs of the neural network on earlier parts of the sequence.""",
            'graphicPath': "/Users/tobiasrothlin/Documents/BachelorArbeit/ScoreClassifier/TransformerPipeline.png",
            'graphicDescription': "Classification Pipeline",
            'dataSet': f"ClassifiedDataSetV1.3 with {folds} folds cross validation",
            'seed': seed,
            'kfolds': folds,
            'opParams': None
        },
        {
            'data': 'Category',
            'model': ZeroShotClassification,
            'modelName': "ZeroShotClassification",
            'modelCreator': "Tobias Rothlin",
            'mlPrinciple': "ZeroShotClassification",
            'refrences': {
                'Text Classification with TF': "https://www.tensorflow.org/text/tutorials/text_classification_rnn"
            },
            'algorithemDescription': """A neural network that is intentionally run multiple times, where parts of each run feed into the next run. Specifically, hidden layers from the previous run provide part of the input to the same hidden layer in the next run. Recurrent neural networks are particularly useful for evaluating sequences, so that the hidden layers can learn from previous runs of the neural network on earlier parts of the sequence.""",
            'graphicPath': "/Users/tobiasrothlin/Documents/BachelorArbeit/ScoreClassifier/TransformerPipeline.png",
            'graphicDescription': "Classification Pipeline",
            'dataSet': f"ClassifiedDataSetV1.3 with {folds} folds cross validation",
            'seed': seed,
            'kfolds': folds,
            'opParams': None
        },
    ]



def modelPerofrmaceEvaluation(data,model,modelName,modelCreator,mlPrinciple,refrences,algorithemDescription,graphicPath,graphicDescription,dataSet,seed,kfolds,opParams):
    if opParams == None:
        opParams = [None]

    for param in opParams:
        random.seed(seed)
        random.shuffle(data)
        if param == None:
             paramForModelName = ""
        else:
            paramForModelName = "_" + str(param)
            print(print(f"{str(param):.^100s}"))


        modelPerformanceReport = ModelReport(modelName+paramForModelName, modelCreator,
                                             mlPrinciple, refrences,
                                             algorithemDescription,
                                             graphicPath, graphicDescription,
                                             dataSet, str(seed))

        for k in range(kfolds):
            testDataStart = int(k * len(data) / kfolds)
            testDataEnd = int(k * len(data) / kfolds) + int(
                len(data) / kfolds)
            testData = data[testDataStart:testDataEnd]
            trainingData = []
            for element, i in zip(data, range(len(data))):
                if not (i >= testDataStart and i < testDataEnd):
                    trainingData.append(element)

            print(
                f"{k}-training({len(trainingData)}/{(len(trainingData) / (len(trainingData) + len(testData))) * 100:.2f}%):Test({len(testData)}/{(len(testData) / (len(trainingData) + len(testData))) * 100:.2f}%) Split completed")

            if testConstants.balancedSplitDataSet:
                trainingData = DataHandler.balanceDataSet(trainingData)

            modelPerformanceReport.addTrainingSet(trainingData)

            print(f"{k}-added training split to performance raport")
            myScoreClassifier = model(trainingData,param=param,debug=True)
            print(f"{k}-model has been trained with training set")
            testResults = []
            trainingResults = []

            for testCase in testData:
                testResults.append(
                    [testCase[1], myScoreClassifier.classify(testCase[0])])

            for testCase in trainingData:
                trainingResults.append(
                    [testCase[1], myScoreClassifier.classify(testCase[0])])

            print(f"{k}-model has been tested\n")
            modelPerformanceReport.addTestResults(testResults)
            modelPerformanceReport.addTrainingResults(trainingResults,myScoreClassifier.getParameters())

        print(f" -creating the model raport")
        modelPerformanceReport.createRaport(modelName+paramForModelName)


if __name__ == '__main__':
    testbenchDataHabler = DataHandler(testConstants.dataLocation)
    #loops through the testConstants dict
    for model in testConstants.modelsToEvaluate[2:3]:
        print("-Loading dataset:")
        if model['data'] == "Score":
            testData = testbenchDataHabler.getScoreData(testConstants.balancedDataSet)
            testData = [data for data in testData if not data[1] == "Neutral"]
        elif model['data'] == "Category":
            testData = testbenchDataHabler.getCategorieData("Location",testConstants.balancedDataSet)
        else:
            testData = []
            print("-Data Source not found!:")
            break

        try:
            print(f"{'Evaluating Model '+ model['modelName']:-^100s}")
            modelPerofrmaceEvaluation(testData,model['model'],model['modelName'],model['modelCreator'],model['mlPrinciple'],model['refrences'],model['algorithemDescription'],model['graphicPath'],model['graphicDescription'],model['dataSet'],model['seed'],model['kfolds'],model['opParams'])
            print(f"\u001b[32m{'Done Evaluating Model '+ model['modelName']:-^100s}")
            print("\u001b[0m")
            print(100*"-")
        except Exception as e:
            print(f"\u001b[31m{'Error During Testing!!':-^100s}")
            print(traceback.format_exc())
            print("\u001b[0m")