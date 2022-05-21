from transformers import AutoTokenizer
import json
import numpy as np

class BertWordTokenizer:

    def __init__(self, filePath):

        self.__maxLenght = 110
        self.__wordToInt = {}
        with open(filePath,"r") as inputFile:
            self.__wordToInt = json.loads(inputFile.read())

        self.__intToWord = {self.__wordToInt[key]:key for key in self.__wordToInt.keys()}

    def tokenizeSentence(self,sentence):
        sentence = sentence.lower()
        att = [1,1]
        toc = [self.__wordToInt["[CLS]"]]
        for word in sentence.split(" "):
            att.append(1)
            if word in self.__wordToInt.keys():
                toc.append(self.__wordToInt[word])
            else:
                toc.append(self.__wordToInt["[UNK]"])
        toc.append(self.__wordToInt["[SEP]"])
        if self.__maxLenght-len(att) < 1:
            print("Warning!!",len(att))
        for i in range(0,self.__maxLenght-len(att)):
            att.append(0)
            toc.append(0)
        return toc,att


    def tokenize(self,sentences):
        tocs = []
        atts = []
        for sentence in sentences:
            toc , att = self.tokenizeSentence(sentence)
            tocs.append(toc)
            atts.append(att)
        return {'input_ids': np.array(tocs), 'attention_mask': np.array(atts)}



if __name__ == '__main__':
    createDataSet = False

    if createDataSet:
        modelName = 'bert-base-uncased'
        toc = AutoTokenizer.from_pretrained(modelName)
        vocab = toc.vocab
        wordTokenizer = {key: vocab[key] for key in vocab.keys() if
                         key[0] != "#"}
        print(wordTokenizer)
        with open("BertTokenizerWordVocab.json", "w",
                  encoding="utf-8") as output:
            output.write(json.dumps(wordTokenizer))
    else:
        myWordTokenizer = BertWordTokenizer("BertTokenizerWordVocab.json")
        print(myWordTokenizer.tokenize(["Tokenizing text is a core task of NLP","This is a test"]))
