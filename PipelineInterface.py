from transformers import pipeline

class PipelineInterface:

    def __init__(self,ModelName):
        self.__pipe = pipeline(task= 'text-classification',model = ModelName)

    def classify(self,sentence):
        res = self.__pipe(sentence)
        return res[0]['label'],res[0]['score']

if __name__ == '__main__':
    myClass = PipelineInterface("Tobias/bert-base-uncased_English_Hotel_sentiment")
    print(myClass.classify("The hotel is very nicely located"))