from re import I
from transformers import TFBertModel, BertConfig, BertTokenizerFast
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical

import keras

from datetime import datetime


class BERT_TextClassification_Production:

    def __init__(self, trainingData=None, **kwargs):

        setOfCats = {sample[1] for sample in trainingData}
        self.__catToInt = {cat: i for i, cat in enumerate(list(setOfCats))}
        self.__intToCat = {self.__catToInt[key]: key for key in
                           self.__catToInt.keys()}

        trainingData = [[sample[0], self.__catToInt[sample[1]]] for sample in
                        trainingData]

        model_name = 'bert-base-uncased'
        self.__max_length = 100

        config = BertConfig.from_pretrained(model_name)
        config.output_hidden_states = False
        self.__tokenizer = BertTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=model_name, config=config)
        transformer_model = TFBertModel.from_pretrained(model_name,
                                                        config=config)

        bert = transformer_model.layers[0]

        input_ids = Input(shape=(self.__max_length,), name='input_ids',
                          dtype='int32')
        inputs = {'input_ids': input_ids}

        bert_model = bert(inputs)[1]
        dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
        pooled_output = dropout(bert_model, training=False)

        category = Dense(units=8, kernel_initializer=TruncatedNormal(
            stddev=config.initializer_range), name='issue')(pooled_output)
        outputs = {'category': category}

        self.__model = Model(inputs=inputs, outputs=outputs,
                             name='BERT_MultiLabel_MultiClass')

        optimizer = Adam(
            learning_rate=5e-05,
            epsilon=1e-08,
            decay=0.01,
            clipnorm=1.0)

        loss = {'category': CategoricalCrossentropy(from_logits=True)}
        metric = {'category': CategoricalAccuracy('accuracy')}

        self.__model.compile(optimizer=optimizer, loss=loss, metrics=metric)

        y_issue = to_categorical([sample[1] for sample in trainingData])

        x = self.__tokenizer(
            text=[sample[0] for sample in trainingData],
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
            x={'input_ids': x['input_ids']},
            y={'category': y_issue},
            validation_split=0.2,
            batch_size=64,
            epochs=10,
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
        for i, value in enumerate(self.__model(senVec)['category'].numpy()[0]):
            if max < value:
                index = i
                max = value
        return self.__intToCat[index]

    def getParameters(self):
        return {'Model': 'bert-base-uncased', 'Tokenizer': 'bert-base-uncased'}


if __name__ == '__main__':
    sen = "The room was very good"
    myScoreClassifier = BERT_TextClassification_Production()
    print(myScoreClassifier.classify(sen))