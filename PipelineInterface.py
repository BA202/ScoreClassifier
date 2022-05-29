from transformers import pipeline,BertConfig
import tensorflow as tf

class PipelineInterface:

    def __init__(self,ModelName,isMultiLabel = True):
        self.__pipe = pipeline(task= 'text-classification',model = ModelName)
        self.__config = BertConfig.from_pretrained(ModelName)
        self.__th = 0
        self.__isMultiLabel = isMultiLabel

    def classify(self,sentence,returnDetailedValues = False):
        if self.__isMultiLabel:
            tok = self.__pipe.tokenizer(sentence, padding=True,truncation=True, return_tensors='tf')
            res = self.__pipe.model(tok)

            conf = tf.nn.softmax(res.logits, axis=-1)
            out = []
            detailedValues = {}
            confidence = 0

            for i, value in enumerate(list(res.logits.numpy()[0])):
                detailedValues[self.__config.id2label[i]] = value
                if value > self.__th:
                    confidence += conf.numpy()[0][i]
                    out.append([self.__config.id2label[i],0,])

            for i in range(len(out)):
                out[i][1] = round(confidence,4)

            if returnDetailedValues:
                return detailedValues
            else:
                return out
        else:
            res = self.__pipe(sentence)
            return [res[0]['label'], res[0]['score']]

if __name__ == '__main__':
    modelName = "Tobias/bert-base-uncased_English_Hotel_sentiment"
    myClass = PipelineInterface(modelName,False)
    from DataHandler.DataHandler import DataHandler
    myDataHander = DataHandler(lan="German")
    myCatData = myDataHander.getCategorieData("Location",multilablel=True)
    temp = {}
    for sample in myCatData:
        if sample[0] in temp.keys():
            temp[sample[0]].add(sample[1])
        else:
            temp[sample[0]] = {sample[1]}

    myCatData = [[key, list(temp[key])] for key in temp.keys()]
    errors = 0
    print(myClass.classify("i really satisfied with this beautiful hotel, friendly staffs, and delicious foods"))
# for i,sample in enumerate(myCatData):
#     if len(sample[1]) > 1:
#         res = myClass.classify(sample[0])
#         errors += 1
#         print(sample[1])
#         print(res,errors)
#         print(100*"-")