from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class Vader:

    def __init__(self,trainingData= None,**kwargs):
        pass


    def classify(self,sentence):
        sid_obj = SentimentIntensityAnalyzer()
        sentiment_dict = sid_obj.polarity_scores(sentence)
        if sentiment_dict['compound'] >= 0.05:
            return "Positive"
        elif sentiment_dict['compound'] <= - 0.05:
            return "Negative"
        else:
            return "Neutral"


    def getParameters(self):
        return None

if __name__ == '__main__':
    sen = "The room was very bad"
    myScoreClassifier = Vader()
    print(myScoreClassifier.classify(sen))
