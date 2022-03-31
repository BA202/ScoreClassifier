from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
import pandas as pd
import tensorflow_datasets as tfds


class BERT_Transformers:

    def __init__(self,trainingData= None,**kwargs):
        self.__tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)

        self.__max_length = 512
        self.__batch_size = 6

        (ds_train, ds_test), ds_info = tfds.load('imdb_reviews',split=(tfds.Split.TRAIN,tfds.Split.TEST),as_supervised=True,with_info=True)

        dataset = tf.data.Dataset.from_generator(lambda: trainingData,str,output_shapes=[None])
        # train dataset
        ds_train_encoded = self.__encode_examples(ds_train).shuffle(10000).batch(self.__batch_size)
        # test dataset
        ds_test_encoded = self.__encode_examples(ds_test).batch(self.__batch_size)


        # recommended learning rate for Adam 5e-5, 3e-5, 2e-5
        self.__learning_rate = 2e-5
        # we will do just 1 epoch, though multiple epochs might be better as long as we will not overfit the model
        self.__number_of_epochs = 1

        self.__model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

        # choosing Adam optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.__learning_rate,
                                             epsilon=1e-08)
        # we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        self.__model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        bert_history = self.__model.fit(ds_train_encoded, epochs=self.__number_of_epochs,
                                 validation_data=ds_test_encoded)


    def __convert_example_to_feature(self,review):
        return self.__tokenizer.encode_plus(review,
                                     add_special_tokens=True,
                                     # add [CLS], [SEP]
                                     max_length=self.__max_length,
                                     # max length of the text that can go to BERT
                                     pad_to_max_length=True,
                                     # add [PAD] tokens
                                     return_attention_mask=True,
                                     # add attention mask to not focus on pad tokens
                                     )

    def __map_example_to_dict(self,input_ids, attention_masks, token_type_ids, label):
        return {
                   "input_ids": input_ids,
                   "token_type_ids": token_type_ids,
                   "attention_mask": attention_masks,
               }, label

    def __encode_examples(self,ds, limit=-1):
        # prepare list, so that we can build up final TensorFlow dataset from slices.
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        label_list = []
        for review, label in tfds.as_numpy(ds):
            bert_input = self.__convert_example_to_feature(review.decode())
            input_ids_list.append(bert_input['input_ids'])
            token_type_ids_list.append(bert_input['token_type_ids'])
            attention_mask_list.append(bert_input['attention_mask'])
            label_list.append([label])
        return tf.data.Dataset.from_tensor_slices((input_ids_list,
                                                   attention_mask_list,
                                                   token_type_ids_list,
                                                   label_list)).map(
            self.__map_example_to_dict)

    def classify(self,sentence):
        tf_batch = self.__tokenizer([sentence], max_length=128, padding=True,
                             truncation=True, return_tensors='tf')
        tf_outputs = self.__model(tf_batch)
        tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
        labels = ['Negative', 'Positive']
        label = tf.argmax(tf_predictions, axis=1)
        label = label.numpy()
        return labels[label[0]]

    def getParameters(self):
        return None


if __name__ == '__main__':
    sen = "The room was very good"
    myScoreClassifier = BERT_Transformers()
    print(myScoreClassifier.classify(sen))