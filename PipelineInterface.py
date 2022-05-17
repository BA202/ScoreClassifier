from transformers import pipeline

class PipelineInterface:

    def __init__(self,ModelName):
        self.__pipe = pipeline(task= 'text-classification',model = ModelName)

    def classify(self,sentence):
        res = self.__pipe(sentence)
        print(res,sentence)
        return res[0]['label'],res[0]['score']

if __name__ == '__main__':
    modelName = "bert-base-uncased_English_MultiLable_classification"
    myClass = PipelineInterface(modelName)
    from DataHandler.DataHandler import DataHandler
    myDataHander = DataHandler()
    myCatData = myDataHander.getCategorieData("Location",multilablel=True)
    temp = {}
    for sample in myCatData:
        if sample[0] in temp.keys():
            temp[sample[0]].add(sample[1])
        else:
            temp[sample[0]] = {sample[1]}

    myCatData = [[key, list(temp[key])] for key in temp.keys()]
    for i,sample in enumerate(myCatData):
        if len(sample[1]) > 1:
            print(sample[1])
            myClass.classify(sample[0])
            print(100*"-")