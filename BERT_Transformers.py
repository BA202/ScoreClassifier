from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
import pandas as pd
import tensorflow_datasets as tfds


class BERT_Transformers:

    def __init__(self,trainingData= None,**kwargs):
        self.__model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
        self.__tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        data = trainingData
        tempData = []
        for frame in data:
            if not frame[1] == "Neutral":
                if frame[1] == "Positive":
                    tempData.append([frame[0], 1])
                else:
                    tempData.append([frame[0], 0])

        data = tempData
        train = pd.DataFrame(data[:-100],
                             columns=['DATA_COLUMN', 'LABEL_COLUMN'])
        test = pd.DataFrame(data[-100:-10],
                            columns=['DATA_COLUMN', 'LABEL_COLUMN'])

        DATA_COLUMN = 'DATA_COLUMN'
        LABEL_COLUMN = 'LABEL_COLUMN'

        train_InputExamples, validation_InputExamples = self.convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN)

        train_data = self.convert_examples_to_tf_dataset(list(train_InputExamples),
                                                    self.__tokenizer)
        train_data = train_data.shuffle(100).batch(32).repeat(2)

        validation_data = self.convert_examples_to_tf_dataset(
            list(validation_InputExamples), self.__tokenizer)
        validation_data = validation_data.batch(32)

        self.__model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5,
                                                         epsilon=1e-08,
                                                         clipnorm=1.0),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy(
                          'accuracy')])

        self.__model.fit(train_data, epochs=2, validation_data=validation_data, )

    def classify(self,sentence):
        tf_batch = self.__tokenizer(sentence, max_length=128, padding=True,truncation=True, return_tensors='tf')
        tf_outputs = self.__model(tf_batch)
        tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
        labels = ['Negative', 'Positive']
        label = tf.argmax(tf_predictions, axis=1)
        label = label.numpy()
        return labels[label[0]]

    def getParameters(self):
        return {'Model':'bert-base-uncased','Tokenizer':'bert-base-uncased'}


    def convert_data_to_examples(self,train, test, DATA_COLUMN, LABEL_COLUMN):
        train_InputExamples = train.apply(
            lambda x: InputExample(guid=None, text_a=x[DATA_COLUMN],
                                   text_b=None, label=x[LABEL_COLUMN]),
            axis=1)
        validation_InputExamples = test.apply(
            lambda x: InputExample(guid=None, text_a=x[DATA_COLUMN],
                                   text_b=None, label=x[LABEL_COLUMN]),
            axis=1)
        return train_InputExamples, validation_InputExamples

    def convert_examples_to_tf_dataset(self,examples, tokenizer,
                                       max_length=128):
        features = []  # -> will hold InputFeatures to be converted later
        for e in examples:
            # Documentation is really strong for this method, so please take a look at it
            input_dict = tokenizer.encode_plus(e.text_a,
                                               add_special_tokens=True,
                                               max_length=max_length,
                                               return_token_type_ids=True,
                                               return_attention_mask=True,
                                               pad_to_max_length=True,
                                               truncation=True)
            input_ids, token_type_ids, attention_mask = (
            input_dict["input_ids"], input_dict["token_type_ids"],
            input_dict['attention_mask'])
            features.append(InputFeatures(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          token_type_ids=token_type_ids,
                                          label=e.label))
        def gen():
            for f in features:
                yield (
                    {
                        "input_ids": f.input_ids,
                        "attention_mask": f.attention_mask,
                        "token_type_ids": f.token_type_ids,
                    },
                    f.label,
                )
        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32,
              "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )


if __name__ == '__main__':
    sen = "The room was very good"
    myScoreClassifier = BERT_Transformers()
    print(myScoreClassifier.classify(sen))