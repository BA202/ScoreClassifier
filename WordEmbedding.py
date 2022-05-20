import numpy as np

class WordEmbedding:

    def __init__(self, soruce):
        self.__wordToVec = {}
        self.__OutputVec = 0
        with open(soruce + "/metadata.tsv", "r") as metaData:
            with open(soruce+ "/vectors.tsv","r") as vecData:
                for word,vecLine in zip(metaData.read().split("\n"),vecData.read().split("\n")):
                    if len(word) > 0:
                        vec = []
                        for value in vecLine.split("\t"):
                            vec.append(float(value))
                        vec = np.array(vec)
                        self.__OutputVec = len(vec)
                        self.__wordToVec[word] = vec

    def toVec(self,word):
        if word.lower() in self.__wordToVec.keys():
            return self.__wordToVec[word.lower()]
        else:
            return self.__wordToVec[list(self.__wordToVec.keys())[0]]

    def vectorize(self,sentence):
        vec = np.zeros((self.__OutputVec))
        for word in sentence.split(" "):
            vec += self.toVec(word)
        return vec


if __name__ == '__main__':
    myTokeniser = WordEmbedding("WordEmbeddings/CBOWEmbedding")

