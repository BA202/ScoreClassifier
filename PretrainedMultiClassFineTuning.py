from transformers import TFBertModel, TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import os
import tensorflow as tf
import keras
from datetime import datetime
import numpy as np

class PretrainedMultiClassFineTuning:

    def __init__(self, trainingData=None, **kwargs):
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

        modelName = 'bert-base-uncased'
        self.__max_length = 100

        setOfCats = list({sample[1] for sample in trainingData})
        self.__catToInt = {cat: i for i, cat in enumerate(list(setOfCats))}
        self.__intToCat = {self.__catToInt[key]: key for key in self.__catToInt.keys()}

        self.__tokenizer = AutoTokenizer.from_pretrained(modelName)
        self.__model = TFAutoModelForSequenceClassification.from_pretrained(modelName,from_pt=True,num_labels=len(setOfCats))

        tfTrainingData = []

        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.metrics.SparseCategoricalAccuracy()
        self.__model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metric)

        tokSens = self.__tokenizer([sample[0] for sample in trainingData],
                                   add_special_tokens=True,
                                   max_length=self.__max_length,
                                   truncation=True,
                                   padding='max_length',
                                   return_tensors='tf',
                                   return_token_type_ids=False,
                                   return_attention_mask=True,
                                   verbose=True)

        logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

        self.__model.fit(
            x=tokSens['input_ids'],
            y=np.array([self.__catToInt[sample[1]] for sample in trainingData]),
            validation_split=0.2,
            batch_size=64,
            epochs=1,
            callbacks=[tensorboard_callback])


    def classify(self, sentence):
        senVec = self.__tokenizer(text=[sentence],
                                  add_special_tokens=True,
                                  max_length=self.__max_length,
                                  truncation=True,
                                  padding='max_length',
                                  return_tensors='tf',
                                  return_token_type_ids=False,
                                  return_attention_mask=False,
                                  verbose=True)
        index = 0
        max = -100
        res = self.__model(senVec)
        for i, value in enumerate(res[0][0]):
            if max < value:
                index = i
                max = value
        return self.__intToCat[index]



    def getParameters(self):
        return {'Epochs':10,'Learning rate':5e-5}