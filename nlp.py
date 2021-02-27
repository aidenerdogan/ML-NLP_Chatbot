import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)

        #add documents in the corpus
        documents.append((w, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# checking results by printing
# print("words: ",words)
# print("classes: ",classes)
# print("documents: ",documents)

# lemmatization: converting a word to its base form
# ‘Caring’ -> Lemmatization -> ‘Care’
# ‘Caring’ -> Stemming -> ‘Car’
# note: not usin POS tagging
# lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# sort classes
classes = sorted(list(set(classes)))

# checking results by printing: while needed you can uncomment print lines
# documents = combination between patterns and intents
# print(len(documents), "documents\n",documents)
# print("==============================\n")
# classes = intents
# print(len(classes), "classes", classes)
# print("==============================\n")
# words = all words, vocabulary
# print(len(words), "unique lemmatized words", words)

# pickling: serializing the objects
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words (bow) for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    # print("training data: ",[bag, output_row]"\n")
    training.append([bag, output_row])

# checking results by printing: while needed you can uncomment print lines
# print(" output: ",output_empty)
