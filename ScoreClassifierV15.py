from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class ScoreClassifierV15:

    def __init__(self,trainingData= None):
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




if __name__ == '__main__':
    sen = "The room was very bad"
    myScoreClassifier = ScoreClassifierV15()
    print(myScoreClassifier.classify(sen))
