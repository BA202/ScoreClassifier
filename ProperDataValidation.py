from MultinomialNaiveBayes import MultinomialNaiveBayes
from Vader import Vader
from SupportVectorMachine_English import SupportVectorMachine
from RandomForest import RandomForest
from RNN import RNN
from ZeroShotClassification import ZeroShotClassification
from TransformesModel import TransformesModel
from DataHandler.DataHandler import DataHandler
from ModelReport.ModelReport import ModelReport
from BERT_Transformers import BERT_Transformers
from BERT_TextClassification_Production import BERT_TextClassification_Production
from PretrainedMultiClassSVM import PretrainedMultiClassSVM
from PretrainedMultiClassFineTuning import PretrainedMultiClassFineTuning
from SupportVectorMachineVectoriserTest import SupportVectorMachineVectoriserTest
from LDAToken import LDAToken
import random
import traceback
import json

class testConstants:
    folds = 10
    seed = 4.83819
    dataLocation = ""
    balancedDataSet = False
    balancedSplitDataSet = False

    modelsToEvaluate = [
        {
            'data': 'Score',
            'model': SupportVectorMachineVectoriserTest,
            'modelName': "Word_TFIDF_SVM",
            'modelCreator': "Tobias Rothlin",
            'mlPrinciple': "WordToVec",
            'refrences': {
                'BertTokenizerFast': "https://huggingface.co/docs/transformers/model_doc/bert"
            },
            'algorithemDescription': """""",
            'graphicPath': "/Users/tobiasrothlin/Documents/BachelorArbeit/ScoreClassifier/TransformerPipeline.png",
            'graphicDescription': "Classification Pipeline",
            'dataSet': f"ClassifiedDataSetV1.4 with {folds} folds cross validation",
            'seed': seed,
            'kfolds': folds,
            'opParams': None
        }
    ]



def modelPerofrmaceEvaluation(trainingData,testData,model,modelName,modelCreator,mlPrinciple,refrences,algorithemDescription,graphicPath,graphicDescription,dataSet,seed,kfolds,opParams):
        modelPerformanceReport = ModelReport(modelName, modelCreator,
                                             mlPrinciple, refrences,
                                             algorithemDescription,
                                             graphicPath, graphicDescription,
                                             dataSet, str(seed))


        modelPerformanceReport.addTrainingSet(trainingData)

        print(f"{0}-added training split to performance raport")
        myScoreClassifier = model(trainingData,debug=True)
        print(f"{0}-model has been trained with training set")
        testResults = []
        trainingResults = []

        for testCase in testData:
            testResults.append(
                [testCase[1], myScoreClassifier.classify(testCase[0])])

        for testCase in trainingData:
            trainingResults.append(
                [testCase[1], myScoreClassifier.classify(testCase[0])])

        print(f"{0}-model has been tested\n")
        modelPerformanceReport.addTestResults(testResults)
        modelPerformanceReport.addTrainingResults(trainingResults,myScoreClassifier.getParameters())

        print(f" -creating the model raport")
        modelPerformanceReport.createRaport(modelName)


if __name__ == '__main__':
    testBenchDataHandlerTraining = DataHandler(testConstants.dataLocation, lan="English")
    testBenchDataHandlerTest = DataHandler("ClassifiedFilesEnglish2.0", lan="English")
    #loops through the testConstants dict
    for model in testConstants.modelsToEvaluate:
        print("-Loading dataset:")
        if model['data'] == "Score":
            testData = testBenchDataHandlerTest.getScoreData(testConstants.balancedDataSet)
            testData = [data for data in testData if not data[1] == "Neutral"]
            trainingData = testBenchDataHandlerTraining.getScoreData(testConstants.balancedDataSet)
            trainingData = [data for data in trainingData if not data[1] == "Neutral"]
        elif model['data'] == "Category":
            testData = testBenchDataHandlerTraining.getCategorieData("Location", testConstants.balancedDataSet)
            trainingData = []
        else:
            testData = []
            trainingData = []
            print("-Data Source not found!:")
            break


        try:
            print(f"{'Evaluating Model '+ model['modelName']:-^100s}")
            modelPerofrmaceEvaluation(trainingData,testData,model['model'],model['modelName'],model['modelCreator'],model['mlPrinciple'],model['refrences'],model['algorithemDescription'],model['graphicPath'],model['graphicDescription'],model['dataSet'],model['seed'],model['kfolds'],model['opParams'])
            print(f"\u001b[32m{'Done Evaluating Model '+ model['modelName']:-^100s}")
            print("\u001b[0m")
            print(100*"-")
        except Exception as e:
            print(f"\u001b[31m{'Error During Testing!!':-^100s}")
            print(traceback.format_exc())
            print("\u001b[0m")