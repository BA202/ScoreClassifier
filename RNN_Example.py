from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
import numpy as np

import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

vocabulary_size = 5000



(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocabulary_size)

newYTrain = []
for data in y_train:
    if data == 1:
        newYTrain.append([1,0])
    else:
        newYTrain.append([0,1])

print(newYTrain)
print('Loaded dataset with {} training samples, {} test samples'.format(len(X_train), len(X_test)))

y_train = np.array(newYTrain)
print('---review---')
print(X_train[6])
print('---label---')
print(y_train[6])

word2id = imdb.get_word_index()
id2word = {i: word for word, i in word2id.items()}
print('---review with words---')
print([id2word.get(i, ' ') for i in X_train[6]])
print('---label---')
print(y_train[6])



print('Maximum review length: {}'.format(
len(max((X_train + X_test), key=len))))

print('Minimum review length: {}'.format(
len(min((X_test + X_test), key=len))))

max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

embedding_size=32
model=Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(LSTM(100))
model.add(Dense(2, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

batch_size = 64
num_epochs = 3

X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]
X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]

model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)
print(X_test[0],y_test[0])
#scores = model.evaluate(X_test, y_test, verbose=0)
#print('Test accuracy:', scores[1])

sen = "this is bad"

test = [word2id[word] for word in sen.split(" ")]
print(test)
test = sequence.pad_sequences([test], maxlen=max_words)

print(sen,model(test))


sen = "this is very good"

test = [word2id[word] for word in sen.split(" ")]
print(test)
test = sequence.pad_sequences([test], maxlen=max_words)

print(sen,model(test))