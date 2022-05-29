from DataHandler.DataHandler import DataHandler
from ModelReport.ModelReport import ModelReport
from PipelineInterface import PipelineInterface




if __name__ == '__main__':
    modelName = "Tobias/bert-base-uncased_English_Hotel_sentiment"
    trainingDataHandler = DataHandler("",lan="English")
    testDataHandler = DataHandler("ClassifiedFilesEnglish2.0", lan="English")
    trainingData = trainingDataHandler.getScoreData()
    testData = testDataHandler.getScoreData()
    pipeline = PipelineInterface(modelName,isMultiLabel=False)
    performanceRapport = ModelReport(modelName=modelName,creatorName="Tobias Rothlin",
                                     MLPrinciple="BERT", dictOfReferences={},
                                     algoDescription="""The BERT model is a pre-trained Encoder trained on masked language modelling. BERT stands for Bidirectional Encoder Representations from Transformers. To train the model, words were randomly masked in a sentence and based on the unmasked words. The model needs to predict the masked word. BERT was trained on a corpus comprising of Books and English Wikipedia containing around 3.3 Billion words.""",
                                     descriptionGraphicPath="/Users/tobiasrothlin/Documents/BachelorArbeit/Bilder/BERT.png",
                                     graphicDescription="Classification Pipeline",
                                     datafile="ClassifiedFilesEnglish2.0",
                                     randomSplitSeed="None")
    performanceRapport.addTrainingSet(trainingData)

    testResults = []
    trainingResults = []

    with open("ErrorFileTraining.tsv", "a") as errorFile:
        errorFile.write(f"Sentence\tTrue\tPredicted\n")

    for sample in trainingData:
        pred = pipeline.classify(sample[0])[0]
        trainingResults.append(
            [sample[1], pred])
        if pred != sample[1]:
            with open("ErrorFileTraining.tsv","a") as errorFile:
                errorFile.write(f"{sample[0]}\t{sample[1]}\t{pred}\n")

    with open("ErrorFileTest.tsv", "a") as errorFile:
        errorFile.write(f"Sentence\tTrue\tPredicted\n")

    for sample in testData:
        pred = pipeline.classify(sample[0])[0]
        testResults.append(
            [sample[1], pred])
        if pred != sample[1]:
            with open("ErrorFileTest.tsv","a") as errorFile:
                errorFile.write(f"{sample[0]}\t{sample[1]}\t{pred}\n")


    performanceRapport.addTrainingResults(trainingResults,{"-":"-"})
    performanceRapport.addTestResults(testResults,)
    performanceRapport.createRaport(modelName,htmlDebug=True)