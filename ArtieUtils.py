from importlib.machinery import WindowsRegistryFinder
import nltk
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer()

def tokenise(sentence):
     return nltk.word_tokenize(sentence)

def stem(words):
    return stemmer.stem(words.lower())

def WordBag(SenToken, all):
    SenToken = [stem(w) for w in SenToken]
    bag = np.zeros(len(all), dtype=np.float32)
    for idx, w in enumerate(all):
        if w in SenToken:
            bag[idx] = 1.0

    return bag
