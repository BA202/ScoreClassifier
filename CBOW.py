import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from DataHandler.DataHandler import DataHandler
import io


runOnBigDataSet = False

sentenceData = []
if runOnBigDataSet:
    with open("Hotel_Reviews_ForEmbedding_Clean.txt") as inputFile:
        for line in inputFile.read().split("\n"):
            sentenceData.append(line)
else:
    myDataHandler = DataHandler()
    sentenceData = [sample[0] for sample in myDataHandler.getScoreData()]


windowSize = 1

print(sentenceData[:3])
listOfContextWindows = []
listOfTargetWords = []

for sentence in sentenceData:
    listOfWordsInSentence = sentence.split(" ")
    for i in range(windowSize,len(listOfWordsInSentence)-windowSize):
        prediction = []
        for j in range(windowSize):
            prediction.append(listOfWordsInSentence[i-1-j])
            prediction.append(listOfWordsInSentence[i+1+j])
        listOfContextWindows.append(" ".join(prediction))
        listOfTargetWords.append(listOfWordsInSentence[i])

for window, target in zip(listOfContextWindows[:3],listOfTargetWords[:3]):
    print(window,target,sep=":")



vocab_size = 10000
sequence_length = windowSize * 2
embedding_dim = 100

vectorize_layer = TextVectorization(
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=None)

vectorize_layer.adapt(sentenceData)


model = Sequential([
  Embedding(vocab_size, embedding_dim, name="embedding"),
  Lambda(lambda x: K.mean(x, axis=1), output_shape=(embedding_dim,)),
  Dense(vocab_size, activation='softmax')
])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
opt = tf.keras.optimizers.Adam(learning_rate=0.005)

model.compile(optimizer=opt,
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
    x = vectorize_layer(listOfContextWindows),
    y = to_categorical(vectorize_layer(listOfTargetWords),num_classes=vocab_size),
    validation_split=0.2,
    batch_size=64,
    epochs=30,
    callbacks=[tensorboard_callback])

model.summary()
weights = model.get_layer('embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()

out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
  if index == 0:
    continue  # skip 0, it's padding.
  vec = weights[index]
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")
out_v.close()
out_m.close()