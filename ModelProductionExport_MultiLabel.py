from transformers import TFBertModel, TFAutoModelForSequenceClassification, TFAutoModelForMultipleChoice
from transformers import AutoTokenizer
import transformers
import os
import tensorflow as tf
import keras
from ModelReport.ModelReport import ModelReport
from DataHandler.DataHandler import DataHandler
import numpy as np
import random
import json
from tensorflow.keras.utils import to_categorical

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

modelName = "bert-base-uncased"
max_length = 100
epochs = 5
learning_rate = 5e-5
seed = 6.838324
language = "English"
path = f"{modelName}_{language}_classification"

productionReport = ModelReport(
    modelName,
    "Tobias Rothlin",
    "Transformer",
    {"bert-base-uncased": "https://huggingface.co/bert-base-uncased"},
    "",
    "",
    "",
    language + " V1.4",
    str(seed),
)


dataHandler = DataHandler("", language)
trainingData = dataHandler.getCategorieData("Location",multilablel=True)
temp = {}
for sample in trainingData:
    if sample[0] in temp.keys():
        temp[sample[0]].add(sample[1])
    else:
        temp[sample[0]] = {sample[1]}

trainingData = [[key,list(temp[key]) ]for key in temp.keys()]
for sample in trainingData:
    pass
    #print(sample[0],sample[1])
#random.seed(seed)
#random.shuffle(trainingData)

testData = trainingData[-100:]
trainingData = trainingData[:-100]

setOfCats = list({sample[1][0] for sample in trainingData})
catToInt = {cat: i for i, cat in enumerate(list(setOfCats))}
intToCat = {catToInt[key]: key for key in catToInt.keys()}

tokenizer = AutoTokenizer.from_pretrained(modelName)
model = TFAutoModelForSequenceClassification.from_pretrained(
    modelName, from_pt=True, num_labels=len(setOfCats), problem_type="multi_label_classification"
)
tfTrainingData = []

productionReport.addTrainingSet(trainingData)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
metric = tf.metrics.SparseCategoricalAccuracy()
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["categorical_accuracy"])
model.summary()
tokSens = tokenizer(
    [sample[0] for sample in trainingData],
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding="max_length",
    return_tensors="tf",
    return_token_type_ids=False,
    return_attention_mask=True,
    verbose=True,
)
temp = []
for sample in trainingData:
    #print(sample)
    temp.append(sum(np.array([to_categorical(catToInt[lbl],num_classes=len(setOfCats)) for lbl in sample[1]])))
y = np.array(temp)

model.fit(
    x=tokSens['input_ids'],
    y=y,
    validation_split=0,
    batch_size=64,
    epochs=epochs,
)


def classify(sentence):
    return intToCat[0]
    senVec = tokenizer(
        text=[sentence],
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="tf",
        return_token_type_ids=False,
        return_attention_mask=False,
        verbose=True,
    )
    index = 0
    max = -100
    res = model(senVec)
    for i, value in enumerate(res[0][0]):
        if max < value:
            index = i
            max = value
    return intToCat[index]


testResults = []
trainingResults = []
for testCase in testData:
    testResults.append([testCase[1], classify(testCase[0])])

for testCase in trainingData:
    trainingResults.append([testCase[1], classify(testCase[0])])

productionReport.addTestResults(testResults)
productionReport.addTrainingResults(
    trainingResults, {"Epochs": str(epochs), "learning_rate": str(learning_rate)}
)

productionReport.createRaport(path)

pipeline = transformers.pipeline(
    "text-classification", model=model, tokenizer=tokenizer
)

# Save pipeline
pipeline.save_pretrained(path)
# Save manifest (needed by OVHcloud ML Serving to load your pipeline)
with open(path + "/manifest.json", "w") as file:
    json.dump(
        {
            "type": "huggingface_pipeline",
            "pipeline_class": type(pipeline).__name__,
            "tokenizer_class": type(pipeline.tokenizer).__name__,
            "model_class": type(pipeline.model).__name__,
        },
        file,
        indent=2,
    )

with open(path + '/intToClass.json', 'w') as file:
    json.dump(intToCat, file, indent=2)