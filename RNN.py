import tensorflow as tf
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
import numpy as np


class RNN:

    def __init__(self,trainingData= None, debugOn = False,isNotPredefined = False,**kwargs):
        self.__trainingData = trainingData
        self.__vocabSize = 5000

        labels = [data[1] for data in trainingData]

        if isNotPredefined:
            self.__LabelToID = {label: i for label,i in zip({lbl for lbl in labels},range(len({lbl for lbl in labels})))}
        else:
            self.__LabelToID = {'Neutral':0,'Positive':1,'Negative':-1}

        self.__IDToLabel = {self.__LabelToID[key]: key for key in self.__LabelToID.keys()}

        if not self.__trainingData == None:
            self.__encoder = tf.keras.layers.TextVectorization(max_tokens=self.__vocabSize,output_sequence_length=194)
            self.__encoder.adapt([data[0] for data in self.__trainingData])


            self.__model= tf.keras.Sequential([
                self.__encoder,
                tf.keras.layers.Embedding(
                    input_dim=len(self.__encoder.get_vocabulary()),
                    output_dim=64,
                    # Use masking to handle the variable sequence lengths
                    mask_zero=True),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            if debugOn:
                print(self.__model.summary())

            self.__model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

            batch_size = 64
            num_epochs = 25
            X_valid, y_valid = np.asarray([data[0] for data in self.__trainingData[:batch_size]]), np.asarray([self.__LabelToID[data[1]] for data in self.__trainingData][:batch_size])
            X_train2, y_train2 = np.asarray([data[0] for data in self.__trainingData[batch_size:]]), np.asarray([self.__LabelToID[data[1]] for data in self.__trainingData][batch_size:])
            self.__model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid),batch_size=batch_size, epochs=num_epochs)


    def strLabelToId(self,str):
        pass


    def classify(self,sentence):
        res = self.__model(np.array([sentence]))
        print(sentence,res)
        return "Positive"


    def cleanUp(self,sen):
        sen = sen.lower()
        return sen

    def getParameters(self):
        return None