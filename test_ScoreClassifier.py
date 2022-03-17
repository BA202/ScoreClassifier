import unittest
from ScoreClassifierV1 import ScoreClassifier
from DataHandler.DataHandler import DataHandler
from ModelReport.ModelReport import ModelReport

import random
from time import process_time

class MyTestCases(unittest.TestCase):
    def test_ValidClassificationToId(self):
        myScoreClassifier = ScoreClassifier()
        self.assertEqual(myScoreClassifier.classToId('Positiv'),1)
        self.assertEqual(myScoreClassifier.classToId('Negative'), -1)
        self.assertEqual(myScoreClassifier.classToId('Neutral'), 0)

    def test_ValidIdToClassification(self):
        myScoreClassifier = ScoreClassifier()
        self.assertEqual(myScoreClassifier.idToClass(1),'Positiv')
        self.assertEqual(myScoreClassifier.idToClass(-1), 'Negative')
        self.assertEqual(myScoreClassifier.idToClass(0), 'Neutral')

    def test_InValidClassificationToId(self):
        myScoreClassifier = ScoreClassifier()
        self.assertRaises(TypeError,myScoreClassifier.classToId,'NoClass')

    def test_CleanUp(self):
        dataHandler = DataHandler()
        fullDataSet = dataHandler.getScoreData()
        myScoreClassifier = ScoreClassifier()
        for data in fullDataSet[:5]:
            print(f"{data[0]}\n{myScoreClassifier.cleanUp(data[0])}")

        self.assertTrue(True)


    def test_ScoreClassifier_FullDataSet(self):
        dataHandler = DataHandler()
        fullDataSet = dataHandler.getCategorieData("Location")

        random.seed(process_time())
        random.shuffle(fullDataSet)

        trainingData = fullDataSet[:-400]
        testData = fullDataSet[-400:]


        modelName = "ScoreClassificationV1"
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

        modelReport = ModelReport(modelName,modelCreator,mlPrinciple,refrences,algorithemDescription,graphicPath,graphicDescription)

        modelReport.addTrainingSet(trainingData)

        myScoreClassifier = ScoreClassifier(trainingData)

        testResult = []
        for testCase in testData:
           testResult.append([testCase[1],myScoreClassifier.classify(testCase[0])])

        modelReport.addTestResults(testResult)
        modelReport.createRaport("LocationClassification")
        self.assertEqual(True, True)

    def test_ScoreClassifiere_BalancedSet(self):
        dataHandler = DataHandler()
        fullDataSet = dataHandler.getScoreData()

        random.seed(process_time())
        random.shuffle(fullDataSet)

        numberOfPositivSampels = 0
        numberOfNegativeSampels = 0
        numberOfNutralSampels = 0
        cutOff = 320
        cutOffMax = 330
        balancedTrainingDataSet = []
        balancedTestDataSet = []
        for sample in fullDataSet:
            if sample[1] == 'Positiv':
                numberOfPositivSampels += 1
                if numberOfPositivSampels <= cutOff:
                    balancedTrainingDataSet.append(sample)
                elif numberOfPositivSampels <= cutOffMax:
                    balancedTestDataSet.append(sample)
            elif sample[1] == 'Neutral':
                numberOfNutralSampels += 1
                if numberOfNutralSampels <= cutOff:
                    balancedTrainingDataSet.append(sample)
                elif numberOfNutralSampels <= cutOffMax:
                    balancedTestDataSet.append(sample)
            elif sample[1] == 'Negative':
                numberOfNegativeSampels += 1
                if numberOfNegativeSampels <= cutOff:
                    balancedTrainingDataSet.append(sample)
                elif numberOfNegativeSampels <= cutOffMax:
                    balancedTestDataSet.append(sample)


        trainingData = balancedTrainingDataSet
        testData = balancedTestDataSet

        modelName = "ScoreClassificationV1"
        modelCreator = "Tobias Rothlin"
        mlPrinciple = "Multinomial Naive Bayes"
        refrences = {
            'NultinomialNB Explained': "https://www.mygreatlearning.com/blog/multinomial-naive-bayes-explained/",
            'Stanford NLP Course': "http://spark-public.s3.amazonaws.com/nlp/slides/naivebayes.pdf",
            'Stanford NLP Lecture': "https://www.youtube.com/playlist?list=PLLssT5z_DsK8HbD2sPcUIDfQ7zmBarMYv",
            'Engilsh Stopwords': "https://www.tutorialspoint.com/python_text_processing/python_remove_stopwords.htm"

        }
        algorithemDescription = """The learning algorithm used in this classification is the Multinomial Naïve Bayes. This approach was chosen as it is easy to implement and is computational very efficient. The first step in the classification pipeline is removing all strop words for example 'i', 'me', 'my', 'myself', etc. A list of English stop word is provided by the nltk module. The stop words remover just removes every word that is in the list of stop words. Next the sentence is passed through the stemmer. Stemmers remove morphological affixes from words, leaving only the word stem. This is done with the PorterStemmer class from the nltk module. The final preprocessing step is to vectorize the sentence. This results in a bag of words representation of the sentence. First all the words must be tokenized and then counted. The result will be a numerical feature vector. To generate this vector the CountVectorizer class from sklearn is used.  This class implements both tokenization and occurrence counting in a single class. With the sentence now represented in a vector the Naïve Bayes classifier can work with this vector. For the implementation of the Naïve Bayes classifier the MultinomialNB class (sklearn) is used. """
        graphicPath = "/Users/tobiasrothlin/Documents/BachelorArbeit/ScoreClassifier/OverviewImg.png"
        graphicDescription = "Classification Pipeline"

        modelReport = ModelReport(modelName, modelCreator, mlPrinciple,
                                  refrences, algorithemDescription,
                                  graphicPath, graphicDescription)

        modelReport.addTrainingSet(trainingData)

        myScoreClassifier = ScoreClassifier(trainingData)

        testResult = []
        for testCase in testData:
            testResult.append(
                [myScoreClassifier.classify(testCase[0]), testCase[1]])

        modelReport.addTestResults(testResult)
        modelReport.createRaport("ScoreClassificationSubSet")
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
