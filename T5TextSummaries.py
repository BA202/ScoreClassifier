from transformers import pipeline
from DataHandler.DataHandler import DataHandler


myDataHandler = DataHandler()
data = myDataHandler.getScoreData()
data = ". ".join([sample[0] for sample in data])
data = " ".join(data.split(" ")[:500])
pipe = pipeline("summarization",model="google/pegasus-cnn_dailymail")
pipe_out = pipe(data)
print(data)
print(pipe_out[0]['summary_text'])
