from SupportVectorMachine import SupportVectorMachine
from DataHandler.DataHandler import DataHandler
from ModelReport.ModelReport import ModelReport
import random
import traceback
import json

class testConstants:
    folds = 10
    seed = 4.83819
    dataLocation = "/Users/tobiasrothlin/Documents/BachelorArbeit/DataSets/ClassifiedDataSetV1.3"
    balancedDataSet = False
    balancedSplitDataSet = False

    modelsToEvaluate = [
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
        }
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
    for model in testConstants.modelsToEvaluate[:1]:
        print("-Loading dataset:")
        if model['data'] == "Score":
            testData = testbenchDataHabler.getScoreData(testConstants.balancedDataSet)
            allLabels = ["Positive","Negative"]
        elif model['data'] == "Category":
            testData = testbenchDataHabler.getCategorieData("Location",testConstants.balancedDataSet)
            allLabels = ["Location","Room","Food","Staff","ReasonForStay", "GeneralUtility","HotelOrganisation"]
        else:
            testData = []
            print("-Data Source not found!:")
            break

        for lbl in allLabels:
            tempData = []
            for dataSample in testData:
                if dataSample[1] == lbl:
                    tempData.append([dataSample[0],dataSample[1]])
                else:
                    tempData.append([dataSample[0], "Not "+ lbl])

            print(tempData)
            try:
                print(f"{'Evaluating Model '+ model['modelName']:-^100s}")
                modelPerofrmaceEvaluation(testData,model['model'],model['modelName']+"_"+lbl,model['modelCreator'],model['mlPrinciple'],model['refrences'],model['algorithemDescription'],model['graphicPath'],model['graphicDescription'],model['dataSet'],model['seed'],model['kfolds'],model['opParams'])
                print(f"\u001b[32m{'Done Evaluating Model '+ model['modelName']:-^100s}")
                print("\u001b[0m")
                print(100*"-")
            except Exception as e:
                print(f"\u001b[31m{'Error During Testing!!':-^100s}")
                print(traceback.format_exc())
                print("\u001b[0m")