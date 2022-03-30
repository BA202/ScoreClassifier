from transformers import pipeline


class TransformesModel:

    def __init__(self,trainingData= None,**kwargs):
        self.__classifier = pipeline('sentiment-analysis')


    def classify(self,sentence):
        res = self.__classifier(sentence)
        if res[0]['label'] == "POSITIVE":
            return 'Positive'
        else:
            return 'Negative'

    def getParameters(self):
        return None



if __name__ == '__main__':
    sen = "The room was very good"
    myScoreClassifier = TransformesModel()
    print(myScoreClassifier.classify(sen))

