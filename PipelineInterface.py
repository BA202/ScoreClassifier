from transformers import pipeline

class PipelineInterface:

    def __init__(self,ModelName):
        self.__pipe = pipeline(task= 'text-classification',model = ModelName)

    def classify(self,sentence):
        res = self.__pipe(sentence)
        print(res)
        return res[0]['label'],res[0]['score']

if __name__ == '__main__':
    modelName = "bert-base-uncased_English_sentiment"
    myClass = PipelineInterface(modelName)
    from DataHandler.DataHandler import DataHandler
    myDataHander = DataHandler()
    myScoreData = myDataHander.getScoreData()
    myScoreData = [sample for sample in myScoreData if not sample[1] == "Neutral"]
    i = 0
    correct = 0
    with open(modelName + "_WrongClassification.tsv", 'w') as outputFile:
        outputFile.write(
            "Sentence" + "\t" + "Prediction" + "\t" + "True" + "\n")
    for sample in myScoreData:
        print(i)
        i += 1
        predicted = myClass.classify(sample[0])
        if predicted[0] == sample[1]:
            correct += 1
        else:
            with open(modelName+"_WrongClassification.tsv",'a') as outputFile:
                outputFile.write(sample[0] + "\t" + predicted[0] + "\t" + sample[1] + "\n")
    print(f"Accuracy: {correct/i}")