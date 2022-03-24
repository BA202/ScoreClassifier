import unittest
from MultinomialNaiveBayes import MultinomialNaiveBayes
from Vader import Vader
from SupportVectorMachine import SupportVectorMachine
from RandomForest import RandomForest
from RNN import RNN
from DataHandler.DataHandler import DataHandler
from ModelReport.ModelReport import ModelReport

import random
from time import process_time


class testConstants:
    folds = 10
    seed = 4.83819
    dataLocation = "/Users/tobiasrothlin/Documents/BachelorArbeit/DataSets/ClassifiedDataSetV1.3"
    balancedDataSet = False
    balancedSplitDataSet = True


class TestbenchClassifier(unittest.TestCase):

    def test_performanceOfVader(self):
        kfolds = testConstants.folds

        modelName = "Vader"
        modelCreator = "Tobias Rothlin"
        mlPrinciple = "Vader Sentiment Analysis"
        refrences = {
            'Vader overview': "https://towardsdatascience.com/sentimental-analysis-using-vader-a3415fef7664",
            'Vader github': "https://github.com/cjhutto/vaderSentiment",

        }
        algorithemDescription = """Vader (Valence Aware Dictionary for sEntiment Reasoning) is a pre trained model used for sentiment analysis. Vader is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. The backbone of Vader is a dictionary that maps lexical features to emotion intensities (sentiment score). To receive the sentiment score of a sentence the intensities of each word are added. For example, words like ‘love’, ‘enjoy’ indicating a positive sentiment. Vader is smart enough to understand basic context like ‘did not love’ as negative. Further it has a basic understanding of capitalization and punctuation to emphasis tone.  Due to this any preprocessing steps should not be done. """
        graphicPath = "/Users/tobiasrothlin/Documents/BachelorArbeit/ScoreClassifier/OverviewImg_Vader.png"
        graphicDescription = "Classification Pipeline"
        dataSet = f"ClassifiedDataSetV1.2 with {kfolds} folds cross validation"
        seed = testConstants.seed

        scoreDataHander = DataHandler(testConstants.dataLocation)
        scoreData = scoreDataHander.getScoreData(testConstants.balancedDataSet)
        random.seed(seed)
        random.shuffle(scoreData)

        modelPerformanceReport = ModelReport(modelName, modelCreator,
                                             mlPrinciple, refrences,
                                             algorithemDescription,
                                             graphicPath, graphicDescription,
                                             dataSet, str(seed))

        for k in range(kfolds):
            testDataStart = int(k * len(scoreData) / kfolds)
            testDataEnd = int(k * len(scoreData) / kfolds) + int(
                len(scoreData) / kfolds)
            testData = scoreData[testDataStart:testDataEnd]
            trainingData = []
            for element, i in zip(scoreData, range(len(scoreData))):
                if not (i >= testDataStart and i < testDataEnd):
                    trainingData.append(element)

            print(
                f"{k}-training({len(trainingData)}/{(len(trainingData) / (len(trainingData) + len(testData))) * 100:.2f}%):Test({len(testData)}/{(len(testData) / (len(trainingData) + len(testData))) * 100:.2f}%) Split completed")

            if testConstants.balancedSplitDataSet:
                trainingData = scoreDataHander.balanceDataSet(trainingData)

            modelPerformanceReport.addTrainingSet(trainingData)
            print(f"{k}-added training split to performance raport")
            myScoreClassifier = Vader(trainingData)
            print(f"{k}-model has been trained with training set")
            testResults = []

            for testCase in testData:
                testResults.append(
                    [testCase[1], myScoreClassifier.classify(testCase[0])])
            print(f"{k}-model has been tested\n\n")
            modelPerformanceReport.addTestResults(testResults)

        print(f"creating the model raport")
        modelPerformanceReport.createRaport("Vader")

        self.assertEqual(True, True)

    def test_performanceOfMultinomialNaiveBayesOnScore(self):
        kfolds = testConstants.folds

        modelName = "MultinomialNaiveBayesOnScore"
        modelCreator = "Tobias Rothlin"
        mlPrinciple = "Multinomial Naive Bayes"
        refrences = {
            'NultinomialNB Explained': "https://towardsdatascience.com/sentimental-analysis-using-vader-a3415fef7664",
            'Stanford NLP Course': "http://spark-public.s3.amazonaws.com/nlp/slides/naivebayes.pdf",
            'Stanford NLP Lecture': "https://www.youtube.com/playlist?list=PLLssT5z_DsK8HbD2sPcUIDfQ7zmBarMYv",
            'Engilsh Stopwords': "https://www.tutorialspoint.com/python_text_processing/python_remove_stopwords.htm"

        }
        algorithemDescription = """The learning algorithm used in this classification is the Multinomial Naïve Bayes. This approach was chosen as it is easy to implement and is computational very efficient. The first step in the classification pipeline is removing all strop words for example 'i', 'me', 'my', 'myself', etc. A list of English stop word is provided by the nltk module. The stop words remover just removes every word that is in the list of stop words. Next the sentence is passed through the stemmer. Stemmers remove morphological affixes from words, leaving only the word stem. This is done with the PorterStemmer class from the nltk module. The final preprocessing step is to vectorize the sentence. This results in a bag of words representation of the sentence. First all the words must be tokenized and then counted. The result will be a numerical feature vector. To generate this vector the CountVectorizer class from sklearn is used.  This class implements both tokenization and occurrence counting in a single class. With the sentence now represented in a vector the Naïve Bayes classifier can work with this vector. For the implementation of the Naïve Bayes classifier the MultinomialNB class (sklearn) is used. """
        graphicPath = "/Users/tobiasrothlin/Documents/BachelorArbeit/ScoreClassifier/OverviewImg.png"
        graphicDescription = "Classification Pipeline"
        dataSet = f"ClassifiedDataSetV1.2 with {kfolds} folds cross validation"
        seed = testConstants.seed

        scoreDataHander = DataHandler(testConstants.dataLocation)
        scoreData = scoreDataHander.getScoreData(testConstants.balancedDataSet)
        random.seed(seed)
        random.shuffle(scoreData)

        modelPerformanceReport = ModelReport(modelName, modelCreator,
                                             mlPrinciple, refrences,
                                             algorithemDescription,
                                             graphicPath, graphicDescription,
                                             dataSet, str(seed))

        for k in range(kfolds):
            testDataStart = int(k * len(scoreData) / kfolds)
            testDataEnd = int(k * len(scoreData) / kfolds) + int(
                len(scoreData) / kfolds)
            testData = scoreData[testDataStart:testDataEnd]
            trainingData = []
            for element, i in zip(scoreData, range(len(scoreData))):
                if not (i >= testDataStart and i < testDataEnd):
                    trainingData.append(element)

            print(
                f"{k}-training({len(trainingData)}/{(len(trainingData) / (len(trainingData) + len(testData))) * 100:.2f}%):Test({len(testData)}/{(len(testData) / (len(trainingData) + len(testData))) * 100:.2f}%) Split completed")

            if testConstants.balancedSplitDataSet:
                trainingData = scoreDataHander.balanceDataSet(trainingData)

            modelPerformanceReport.addTrainingSet(trainingData)
            print(f"{k}-added training split to performance raport")
            myScoreClassifier = MultinomialNaiveBayes(trainingData)
            print(f"{k}-model has been trained with training set")
            testResults = []

            for testCase in testData:
                testResults.append(
                    [testCase[1], myScoreClassifier.classify(testCase[0])])
            print(f"{k}-model has been tested\n\n")
            modelPerformanceReport.addTestResults(testResults)

        print(f"creating the model raport")
        modelPerformanceReport.createRaport("MultinomialNaiveBayesOnScore")

        self.assertEqual(True, True)

    def test_performanceOfMultinomialNaiveBayesOnLocation(self):
        kfolds = testConstants.folds

        modelName = "MultinomialNaiveBayesOnLocation"
        modelCreator = "Tobias Rothlin"
        mlPrinciple = "Multinomial Naive Bayes"
        refrences = {
            'NultinomialNB Explained': "https://towardsdatascience.com/sentimental-analysis-using-vader-a3415fef7664",
            'Stanford NLP Course': "http://spark-public.s3.amazonaws.com/nlp/slides/naivebayes.pdf",
            'Stanford NLP Lecture': "https://www.youtube.com/playlist?list=PLLssT5z_DsK8HbD2sPcUIDfQ7zmBarMYv",
            'Engilsh Stopwords': "https://www.tutorialspoint.com/python_text_processing/python_remove_stopwords.htm"

        }
        algorithemDescription = """The learning algorithm used in this classification is the Multinomial Naïve Bayes. This approach was chosen as it is easy to implement and is computational very efficient. The first step in the classification pipeline is removing all strop words for example 'i', 'me', 'my', 'myself', etc. A list of English stop word is provided by the nltk module. The stop words remover just removes every word that is in the list of stop words. Next the sentence is passed through the stemmer. Stemmers remove morphological affixes from words, leaving only the word stem. This is done with the PorterStemmer class from the nltk module. The final preprocessing step is to vectorize the sentence. This results in a bag of words representation of the sentence. First all the words must be tokenized and then counted. The result will be a numerical feature vector. To generate this vector the CountVectorizer class from sklearn is used.  This class implements both tokenization and occurrence counting in a single class. With the sentence now represented in a vector the Naïve Bayes classifier can work with this vector. For the implementation of the Naïve Bayes classifier the MultinomialNB class (sklearn) is used. """
        graphicPath = "/Users/tobiasrothlin/Documents/BachelorArbeit/ScoreClassifier/OverviewImg.png"
        graphicDescription = "Classification Pipeline"
        dataSet = f"ClassifiedDataSetV1.2 with {kfolds} folds cross validation"
        seed = testConstants.seed

        scoreDataHander = DataHandler(testConstants.dataLocation)
        scoreData = scoreDataHander.getCategorieData("Location",testConstants.balancedDataSet)
        random.seed(seed)
        random.shuffle(scoreData)

        modelPerformanceReport = ModelReport(modelName, modelCreator,
                                             mlPrinciple, refrences,
                                             algorithemDescription,
                                             graphicPath, graphicDescription,
                                             dataSet, str(seed))

        for k in range(kfolds):
            testDataStart = int(k * len(scoreData) / kfolds)
            testDataEnd = int(k * len(scoreData) / kfolds) + int(
                len(scoreData) / kfolds)
            testData = scoreData[testDataStart:testDataEnd]
            trainingData = []
            for element, i in zip(scoreData, range(len(scoreData))):
                if not (i >= testDataStart and i < testDataEnd):
                    trainingData.append(element)

            print(
                f"{k}-training({len(trainingData)}/{(len(trainingData) / (len(trainingData) + len(testData))) * 100:.2f}%):Test({len(testData)}/{(len(testData) / (len(trainingData) + len(testData))) * 100:.2f}%) Split completed")

            if testConstants.balancedSplitDataSet:
                trainingData = scoreDataHander.balanceDataSet(trainingData)

            modelPerformanceReport.addTrainingSet(trainingData)
            print(f"{k}-added training split to performance raport")
            myScoreClassifier = MultinomialNaiveBayes(trainingData)
            print(f"{k}-model has been trained with training set")
            testResults = []

            for testCase in testData:
                testResults.append(
                    [testCase[1], myScoreClassifier.classify(testCase[0])])
            print(f"{k}-model has been tested\n\n")
            modelPerformanceReport.addTestResults(testResults)

        print(f"creating the model raport")
        modelPerformanceReport.createRaport("MultinomialNaiveBayesOnLocation")

        self.assertEqual(True, True)

    def test_performanceOfMultinomialNaiveBayesOnScoreWithPosAndNeg(self):
        kfolds = testConstants.folds

        modelName = "MultinomialNaiveBayesOnScore"
        modelCreator = "Tobias Rothlin"
        mlPrinciple = "Multinomial Naive Bayes"
        refrences = {
            'NultinomialNB Explained': "https://towardsdatascience.com/sentimental-analysis-using-vader-a3415fef7664",
            'Stanford NLP Course': "http://spark-public.s3.amazonaws.com/nlp/slides/naivebayes.pdf",
            'Stanford NLP Lecture': "https://www.youtube.com/playlist?list=PLLssT5z_DsK8HbD2sPcUIDfQ7zmBarMYv",
            'Engilsh Stopwords': "https://www.tutorialspoint.com/python_text_processing/python_remove_stopwords.htm"

        }
        algorithemDescription = """The learning algorithm used in this classification is the Multinomial Naïve Bayes. This approach was chosen as it is easy to implement and is computational very efficient. The first step in the classification pipeline is removing all strop words for example 'i', 'me', 'my', 'myself', etc. A list of English stop word is provided by the nltk module. The stop words remover just removes every word that is in the list of stop words. Next the sentence is passed through the stemmer. Stemmers remove morphological affixes from words, leaving only the word stem. This is done with the PorterStemmer class from the nltk module. The final preprocessing step is to vectorize the sentence. This results in a bag of words representation of the sentence. First all the words must be tokenized and then counted. The result will be a numerical feature vector. To generate this vector the CountVectorizer class from sklearn is used.  This class implements both tokenization and occurrence counting in a single class. With the sentence now represented in a vector the Naïve Bayes classifier can work with this vector. For the implementation of the Naïve Bayes classifier the MultinomialNB class (sklearn) is used. """
        graphicPath = "/Users/tobiasrothlin/Documents/BachelorArbeit/ScoreClassifier/OverviewImg.png"
        graphicDescription = "Classification Pipeline"
        dataSet = f"ClassifiedDataSetV1.2 with {kfolds} folds cross validation"
        seed = testConstants.seed

        scoreDataHander = DataHandler(testConstants.dataLocation)
        scoreData = scoreDataHander.getScoreData(testConstants.balancedDataSet)
        scoreData = [data for data in scoreData if not data[1] == "Neutral"]
        random.seed(seed)
        random.shuffle(scoreData)

        modelPerformanceReport = ModelReport(modelName, modelCreator,
                                             mlPrinciple, refrences,
                                             algorithemDescription,
                                             graphicPath, graphicDescription,
                                             dataSet, str(seed))

        for k in range(kfolds):
            testDataStart = int(k * len(scoreData) / kfolds)
            testDataEnd = int(k * len(scoreData) / kfolds) + int(
                len(scoreData) / kfolds)
            testData = scoreData[testDataStart:testDataEnd]
            trainingData = []
            for element, i in zip(scoreData, range(len(scoreData))):
                if not (i >= testDataStart and i < testDataEnd):
                    trainingData.append(element)

            print(
                f"{k}-training({len(trainingData)}/{(len(trainingData) / (len(trainingData) + len(testData))) * 100:.2f}%):Test({len(testData)}/{(len(testData) / (len(trainingData) + len(testData))) * 100:.2f}%) Split completed")

            if testConstants.balancedSplitDataSet:
                trainingData = scoreDataHander.balanceDataSet(trainingData)

            modelPerformanceReport.addTrainingSet(trainingData)
            print(f"{k}-added training split to performance raport")
            myScoreClassifier = MultinomialNaiveBayes(trainingData)
            print(f"{k}-model has been trained with training set")
            testResults = []

            for testCase in testData:
                testResults.append(
                    [testCase[1], myScoreClassifier.classify(testCase[0])])
            print(f"{k}-model has been tested\n\n")
            modelPerformanceReport.addTestResults(testResults)

        print(f"creating the model raport")
        modelPerformanceReport.createRaport(
            "MultinomialNaiveBayesOnScoreWithPosAndNeg")

        self.assertEqual(True, True)

    def test_performanceOfSupportVectorMachineOnScore(self):
        listOfKernels = ['poly','rbf','linear', 'sigmoid']
        for kernel in listOfKernels:
            self.performanceOfSupportVectorMachineLinearOnScore(kernel)
            self.performanceOfSupportVectorMachineLinearOnLocation(kernel)
        self.assertEqual(True, True)

    # Support method to minimize code duplication
    def performanceOfSupportVectorMachineLinearOnScore(self, kernel):
        kfolds = testConstants.folds

        modelName = f"SupportVectorMachineOnScore {kernel}"
        modelCreator = "Tobias Rothlin"
        mlPrinciple = "Support Vector Machine"
        refrences = {
            'Sentiment Analysis SVM': "https://medium.com/@vasista/sentiment-analysis-using-svm-338d418e3ff1",
            'Scikit SVM Kernels': "https://scikit-learn.org/stable/modules/svm.html#svm-kernels",
            'Scikit feature extraction': "https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction",
            'Scikit Vectorizer': "https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html"
        }
        algorithemDescription = f"""Support vector machines are a robust supervised learning model based on statistical learning. The idea is to find a Hyperplane separating the different classes with the most separation between the closest points. Before the SVM can classify a sentence, the sentence needs to be vectorised. To accomplish the Scikit learn, Tfidf Vectorizer is used. The Vectorizer converts the sentence to a fixed feature vector. With the vectorised sentences, the model can be trained. The best hyperplanes are found in the training step based on the training data. The flexibility of the hyperplane can be defined by the Kernel (linear, sigmoid, RBF). RBF is used for non-linear problems and is also a general-purpose kernel. This model uses a {kernel} kernel."""
        graphicPath = "/Users/tobiasrothlin/Documents/BachelorArbeit/ScoreClassifier/SVMPipeline.png"
        graphicDescription = "Classification Pipeline"
        dataSet = f"ClassifiedDataSetV1.2 with {kfolds} folds cross validation"
        seed = testConstants.seed

        scoreDataHander = DataHandler(testConstants.dataLocation)
        scoreData = scoreDataHander.getScoreData(testConstants.balancedDataSet)
        random.seed(seed)
        random.shuffle(scoreData)

        modelPerformanceReport = ModelReport(modelName, modelCreator,
                                             mlPrinciple, refrences,
                                             algorithemDescription,
                                             graphicPath, graphicDescription,
                                             dataSet, str(seed))

        for k in range(kfolds):
            testDataStart = int(k * len(scoreData) / kfolds)
            testDataEnd = int(k * len(scoreData) / kfolds) + int(
                len(scoreData) / kfolds)
            testData = scoreData[testDataStart:testDataEnd]
            trainingData = []
            for element, i in zip(scoreData, range(len(scoreData))):
                if not (i >= testDataStart and i < testDataEnd):
                    trainingData.append(element)

            print(
                f"{k}-training({len(trainingData)}/{(len(trainingData) / (len(trainingData) + len(testData))) * 100:.2f}%):Test({len(testData)}/{(len(testData) / (len(trainingData) + len(testData))) * 100:.2f}%) Split completed")

            if testConstants.balancedSplitDataSet:
                trainingData = scoreDataHander.balanceDataSet(trainingData)

            modelPerformanceReport.addTrainingSet(trainingData)
            print(f"{k}-added training split to performance raport")
            myScoreClassifier = SupportVectorMachine(trainingData, kernel)
            print(f"{k}-model has been trained with training set")
            testResults = []

            for testCase in testData:
                testResults.append(
                    [testCase[1], myScoreClassifier.classify(testCase[0])])
            print(f"{k}-model has been tested\n\n")
            modelPerformanceReport.addTestResults(testResults)

        print(f"creating the model raport")
        modelPerformanceReport.createRaport(
            f"SupportVectorMachineOnScore_{kernel}")

    # Support method to minimize code duplication
    def performanceOfSupportVectorMachineLinearOnLocation(self, kernel):
        kfolds = testConstants.folds

        modelName = f"SupportVectorMachineOnLocation {kernel}"
        modelCreator = "Tobias Rothlin"
        mlPrinciple = "Support Vector Machine"
        refrences = {
            'Sentiment Analysis SVM': "https://medium.com/@vasista/sentiment-analysis-using-svm-338d418e3ff1",
            'Scikit SVM Kernels': "https://scikit-learn.org/stable/modules/svm.html#svm-kernels",
            'Scikit feature extraction': "https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction",
            'Scikit Vectorizer': "https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html"
        }
        algorithemDescription = f"""Support vector machines are a robust supervised learning model based on statistical learning. The idea is to find a Hyperplane separating the different classes with the most separation between the closest points. Before the SVM can classify a sentence, the sentence needs to be vectorised. To accomplish the Scikit learn, Tfidf Vectorizer is used. The Vectorizer converts the sentence to a fixed feature vector. With the vectorised sentences, the model can be trained. The best hyperplanes are found in the training step based on the training data. The flexibility of the hyperplane can be defined by the Kernel (linear, sigmoid, RBF). RBF is used for non-linear problems and is also a general-purpose kernel. This model uses a {kernel} kernel."""
        graphicPath = "/Users/tobiasrothlin/Documents/BachelorArbeit/ScoreClassifier/SVMPipeline.png"
        graphicDescription = "Classification Pipeline"
        dataSet = f"ClassifiedDataSetV1.2 with {kfolds} folds cross validation"
        seed = testConstants.seed

        scoreDataHander = DataHandler(testConstants.dataLocation)
        scoreData = scoreDataHander.getCategorieData("Location",testConstants.balancedDataSet)
        random.seed(seed)
        random.shuffle(scoreData)

        modelPerformanceReport = ModelReport(modelName, modelCreator,
                                             mlPrinciple, refrences,
                                             algorithemDescription,
                                             graphicPath, graphicDescription,
                                             dataSet, str(seed))

        for k in range(kfolds):
            testDataStart = int(k * len(scoreData) / kfolds)
            testDataEnd = int(k * len(scoreData) / kfolds) + int(
                len(scoreData) / kfolds)
            testData = scoreData[testDataStart:testDataEnd]
            trainingData = []
            for element, i in zip(scoreData, range(len(scoreData))):
                if not (i >= testDataStart and i < testDataEnd):
                    trainingData.append(element)

            print(
                f"{k}-training({len(trainingData)}/{(len(trainingData) / (len(trainingData) + len(testData))) * 100:.2f}%):Test({len(testData)}/{(len(testData) / (len(trainingData) + len(testData))) * 100:.2f}%) Split completed")

            if testConstants.balancedSplitDataSet:
                trainingData = scoreDataHander.balanceDataSet(trainingData)

            modelPerformanceReport.addTrainingSet(trainingData)
            print(f"{k}-added training split to performance raport")
            myScoreClassifier = SupportVectorMachine(trainingData, kernel)
            print(f"{k}-model has been trained with training set")
            testResults = []

            for testCase in testData:
                testResults.append(
                    [testCase[1], myScoreClassifier.classify(testCase[0])])
            print(f"{k}-model has been tested\n\n")
            modelPerformanceReport.addTestResults(testResults)

        print(f"creating the model raport")
        modelPerformanceReport.createRaport(
            f"SupportVectorMachineOnLocation_{kernel}")

    def test_performanceOfRandomForestOnScore(self):
        kfolds = testConstants.folds

        modelName = "RandomForestOnScore"
        modelCreator = "Tobias Rothlin"
        mlPrinciple = "Random Forest"
        refrences = {
            'NultinomialNB Explained': "https://towardsdatascience.com/sentimental-analysis-using-vader-a3415fef7664",
            'Stanford NLP Course': "http://spark-public.s3.amazonaws.com/nlp/slides/naivebayes.pdf",
            'Stanford NLP Lecture': "https://www.youtube.com/playlist?list=PLLssT5z_DsK8HbD2sPcUIDfQ7zmBarMYv",
            'Engilsh Stopwords': "https://www.tutorialspoint.com/python_text_processing/python_remove_stopwords.htm"

        }
        algorithemDescription = """The learning algorithm used in this classification is the Multinomial Naïve Bayes. This approach was chosen as it is easy to implement and is computational very efficient. The first step in the classification pipeline is removing all strop words for example 'i', 'me', 'my', 'myself', etc. A list of English stop word is provided by the nltk module. The stop words remover just removes every word that is in the list of stop words. Next the sentence is passed through the stemmer. Stemmers remove morphological affixes from words, leaving only the word stem. This is done with the PorterStemmer class from the nltk module. The final preprocessing step is to vectorize the sentence. This results in a bag of words representation of the sentence. First all the words must be tokenized and then counted. The result will be a numerical feature vector. To generate this vector the CountVectorizer class from sklearn is used.  This class implements both tokenization and occurrence counting in a single class. With the sentence now represented in a vector the Naïve Bayes classifier can work with this vector. For the implementation of the Naïve Bayes classifier the MultinomialNB class (sklearn) is used. """
        graphicPath = "/Users/tobiasrothlin/Documents/BachelorArbeit/ScoreClassifier/OverviewImg.png"
        graphicDescription = "Classification Pipeline"
        dataSet = f"ClassifiedDataSetV1.2 with {kfolds} folds cross validation"
        seed = testConstants.seed

        scoreDataHander = DataHandler(testConstants.dataLocation)
        scoreData = scoreDataHander.getScoreData(testConstants.balancedDataSet)
        random.seed(seed)
        random.shuffle(scoreData)

        modelPerformanceReport = ModelReport(modelName, modelCreator,
                                             mlPrinciple, refrences,
                                             algorithemDescription,
                                             graphicPath, graphicDescription,
                                             dataSet, str(seed))

        for k in range(kfolds):
            testDataStart = int(k * len(scoreData) / kfolds)
            testDataEnd = int(k * len(scoreData) / kfolds) + int(
                len(scoreData) / kfolds)
            testData = scoreData[testDataStart:testDataEnd]
            trainingData = []
            for element, i in zip(scoreData, range(len(scoreData))):
                if not (i >= testDataStart and i < testDataEnd):
                    trainingData.append(element)

            print(
                f"{k}-training({len(trainingData)}/{(len(trainingData) / (len(trainingData) + len(testData))) * 100:.2f}%):Test({len(testData)}/{(len(testData) / (len(trainingData) + len(testData))) * 100:.2f}%) Split completed")

            if testConstants.balancedSplitDataSet:
                trainingData = scoreDataHander.balanceDataSet(trainingData)

            modelPerformanceReport.addTrainingSet(trainingData)
            print(f"{k}-added training split to performance raport")
            myScoreClassifier = RandomForest(trainingData)
            print(f"{k}-model has been trained with training set")
            testResults = []

            for testCase in testData:
                testResults.append(
                    [testCase[1], myScoreClassifier.classify(testCase[0])])
            print(f"{k}-model has been tested\n\n")
            modelPerformanceReport.addTestResults(testResults)

        print(f"creating the model raport")
        modelPerformanceReport.createRaport(
            "RandomForestOnScore")

        self.assertEqual(True, True)

    def test_performanceOfRandomForestOnLocation(self):
        kfolds = testConstants.folds

        modelName = "RandomForestOnLocation"
        modelCreator = "Tobias Rothlin"
        mlPrinciple = "Random Forest"
        refrences = {
            'NultinomialNB Explained': "https://towardsdatascience.com/sentimental-analysis-using-vader-a3415fef7664",
            'Stanford NLP Course': "http://spark-public.s3.amazonaws.com/nlp/slides/naivebayes.pdf",
            'Stanford NLP Lecture': "https://www.youtube.com/playlist?list=PLLssT5z_DsK8HbD2sPcUIDfQ7zmBarMYv",
            'Engilsh Stopwords': "https://www.tutorialspoint.com/python_text_processing/python_remove_stopwords.htm"

        }
        algorithemDescription = """The learning algorithm used in this classification is the Multinomial Naïve Bayes. This approach was chosen as it is easy to implement and is computational very efficient. The first step in the classification pipeline is removing all strop words for example 'i', 'me', 'my', 'myself', etc. A list of English stop word is provided by the nltk module. The stop words remover just removes every word that is in the list of stop words. Next the sentence is passed through the stemmer. Stemmers remove morphological affixes from words, leaving only the word stem. This is done with the PorterStemmer class from the nltk module. The final preprocessing step is to vectorize the sentence. This results in a bag of words representation of the sentence. First all the words must be tokenized and then counted. The result will be a numerical feature vector. To generate this vector the CountVectorizer class from sklearn is used.  This class implements both tokenization and occurrence counting in a single class. With the sentence now represented in a vector the Naïve Bayes classifier can work with this vector. For the implementation of the Naïve Bayes classifier the MultinomialNB class (sklearn) is used. """
        graphicPath = "/Users/tobiasrothlin/Documents/BachelorArbeit/ScoreClassifier/OverviewImg.png"
        graphicDescription = "Classification Pipeline"
        dataSet = f"ClassifiedDataSetV1.2 with {kfolds} folds cross validation"
        seed = testConstants.seed

        scoreDataHander = DataHandler(testConstants.dataLocation)
        scoreData = scoreDataHander.getCategorieData("Location",testConstants.balancedDataSet)
        random.seed(seed)
        random.shuffle(scoreData)

        modelPerformanceReport = ModelReport(modelName, modelCreator,
                                             mlPrinciple, refrences,
                                             algorithemDescription,
                                             graphicPath, graphicDescription,
                                             dataSet, str(seed))

        for k in range(kfolds):
            testDataStart = int(k * len(scoreData) / kfolds)
            testDataEnd = int(k * len(scoreData) / kfolds) + int(
                len(scoreData) / kfolds)
            testData = scoreData[testDataStart:testDataEnd]
            trainingData = []
            for element, i in zip(scoreData, range(len(scoreData))):
                if not (i >= testDataStart and i < testDataEnd):
                    trainingData.append(element)

            print(
                f"{k}-training({len(trainingData)}/{(len(trainingData) / (len(trainingData) + len(testData))) * 100:.2f}%):Test({len(testData)}/{(len(testData) / (len(trainingData) + len(testData))) * 100:.2f}%) Split completed")

            if testConstants.balancedSplitDataSet:
                trainingData = scoreDataHander.balanceDataSet(trainingData)

            modelPerformanceReport.addTrainingSet(trainingData)
            print(f"{k}-added training split to performance raport")
            myScoreClassifier = RandomForest(trainingData)
            print(f"{k}-model has been trained with training set")
            testResults = []

            for testCase in testData:
                testResults.append(
                    [testCase[1], myScoreClassifier.classify(testCase[0])])
            print(f"{k}-model has been tested\n\n")
            modelPerformanceReport.addTestResults(testResults)

        print(f"creating the model raport")
        modelPerformanceReport.createRaport(
            "RandomForestOnLocation")

        self.assertEqual(True, True)

    def test_performanceOfRNNOnScore(self):
        kfolds = testConstants.folds

        modelName = "RNNOnScore"
        modelCreator = "Tobias Rothlin"
        mlPrinciple = "RNN"
        refrences = {
            'NultinomialNB Explained': "https://towardsdatascience.com/sentimental-analysis-using-vader-a3415fef7664",

        }
        algorithemDescription = """"""
        graphicPath = "/Users/tobiasrothlin/Documents/BachelorArbeit/ScoreClassifier/OverviewImg.png"
        graphicDescription = "Classification Pipeline"
        dataSet = f"ClassifiedDataSetV1.2 with {kfolds} folds cross validation"
        seed = testConstants.seed

        scoreDataHander = DataHandler(testConstants.dataLocation)
        scoreData = scoreDataHander.getScoreData(testConstants.balancedDataSet)
        random.seed(seed)
        random.shuffle(scoreData)

        modelPerformanceReport = ModelReport(modelName, modelCreator,
                                             mlPrinciple, refrences,
                                             algorithemDescription,
                                             graphicPath, graphicDescription,
                                             dataSet, str(seed))

        for k in range(kfolds):
            testDataStart = int(k * len(scoreData) / kfolds)
            testDataEnd = int(k * len(scoreData) / kfolds) + int(
                len(scoreData) / kfolds)
            testData = scoreData[testDataStart:testDataEnd]
            trainingData = []
            for element, i in zip(scoreData, range(len(scoreData))):
                if not (i >= testDataStart and i < testDataEnd):
                    trainingData.append(element)

            print(
                f"{k}-training({len(trainingData)}/{(len(trainingData) / (len(trainingData) + len(testData))) * 100:.2f}%):Test({len(testData)}/{(len(testData) / (len(trainingData) + len(testData))) * 100:.2f}%) Split completed")

            if testConstants.balancedSplitDataSet:
                trainingData = scoreDataHander.balanceDataSet(trainingData)

            modelPerformanceReport.addTrainingSet(trainingData)
            print(f"{k}-added training split to performance raport")
            myScoreClassifier = RNN(trainingData)
            print(f"{k}-model has been trained with training set")
            testResults = []

            for testCase in testData:
                testResults.append(
                    [testCase[1], myScoreClassifier.classify(testCase[0])])
            print(f"{k}-model has been tested\n\n")
            modelPerformanceReport.addTestResults(testResults)

        print(f"creating the model raport")
        modelPerformanceReport.createRaport("RNNOnScore")

        self.assertEqual(True, True)

if __name__ == '__main__':
    unittest.main()
