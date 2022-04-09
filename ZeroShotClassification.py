from transformers import pipeline
import numpy as np

class ZeroShotClassification:

    def __init__(self,trainingData= None,**kwargs):
        self.__classifier = pipeline("zero-shot-classification",
                              model="facebook/bart-large-mnli")

        self.__labels = ["Location","Room","Food","Staff","ReasonForStay", "GeneralUtility","HotelOrganisation", "Unknown"]

        self.__hypothesis_template = "The review is about {}."
        print(self.__classifier)


    def classify(self,sentence):
        if len(sentence) > 0:
            prediction = self.__classifier(sentence, self.__labels,
                                    hypothesis_template=self.__hypothesis_template,
                                    multi_label=True)
            category = prediction['labels'][np.argmax(prediction['scores'])]
            score = max(prediction['scores'])
            print(f"{sentence:100}:{category:20}:{score*100:.2f}%")
            return category
        else:
            return self.__labels[0]

    def getParameters(self):
        return {'Model': 'facebook/bart-large-mnli'}
