import nltk
from DataHandler.DataHandler import DataHandler
from langid.langid import LanguageIdentifier, model
from spacy_langdetect import LanguageDetector
from spacy.language import Language
import spacy

def get_lang_detector(nlp, name):
    return LanguageDetector()



import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context


nlp = spacy.load("en_core_web_sm")
Language.factory("language_detector", func=get_lang_detector)
nlp.add_pipe('language_detector', last=True)



nltk.download('crubadan')
nltk.download('punkt')



def detect_language_Spacy(sen):
    doc = nlp(sen)
    detect_language = doc._.language
    shortHandToCustomLable= {
        'de':'deu',
        'en':'eng'
    }
    if detect_language['language'] == "en":
        return 'eng'
    elif detect_language['language'] == "de":
        return 'deu'
    else:
        return 'something else'

def detect_language_langid(text):
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    language = identifier.classify(text)
    if language[0] == "en":
        return 'eng'
    elif language[0] == "de":
        return 'deu'
    else:
        return 'something else'

myDataHndler = DataHandler()
scoreData = myDataHndler.getScoreData()

tc = nltk.classify.textcat.TextCat()

spacyError = 0
langidError = 0

for i, sample in enumerate(scoreData):

    if i%100 == 0:
        print(f"{i}-spacyError:{spacyError} langidError:{langidError}")
    else:
        print(i)
    if not detect_language_langid(sample[0]) == 'eng':
        langidError += 1
    if not detect_language_Spacy(sample[0]) == 'eng':
        spacyError += 1


print(f"spacyError:{spacyError} langidError:{langidError}")



