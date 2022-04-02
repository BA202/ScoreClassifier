from transformers import pipeline

class SentimentAnalysisPipeline:

    def __init__(self,trainingData= None,**kwargs):
        self.__pipe = pipeline(task= 'sentiment-analysis',model = 'TrainedModel_BERT_SentimentAnalysis_HotelReviews')


    def classify(self,sentence):
        res = self.__pipe(sentence)
        if res[0]['label'] == 'LABEL_0':
            label = "Negative"
        else:
            label = "Positive"

        return (label, res[0]['score']) 


    def getParameters(self):
        return {'Layers': "3",'Type': "TFBertMainLayer(109'482'240), Dropout(0),Dense(1538)"}


if __name__ == '__main__':
    myClass = SentimentAnalysisPipeline()
    print(myClass.classify("The movie was very bad"))