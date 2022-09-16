#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 21:40:11 2022

@author: Nay Zaw Aung Win
"""

# third file of addiction study project
# will try to do text generation with RNN in this file


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import string
import re
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
import pickle
# import hdbscan


# not removing stop words in the clean text function because RNN might better learn the semantic struture with them
def cleanText(df, col, save_2_file=False):  # this function will take in a data frame, then combine the txt and clean them
    text = ""
    df_txt_lst = df[col].tolist()
    for txt in df_txt_lst:
        text += txt + " "
    text = ' '.join(text.strip().split())
    with open('test_compare.txt', 'w+') as f:
        f.write(text)
    text = re.sub('http*\S+', ' ', text) # remove the urls
    text = text.encode('ascii', 'ignore').decode() # removing the unicode characters like the emojis
    text = text.replace('-', ' ')
    punc_map = str.maketrans('', '', string.punctuation) # whenever, we see a punctuation, replace it with empty string
    tokens = text.split()
    tokens = [tk.translate(punc_map) for tk in tokens]
    tokens = [tk.lower() for tk in tokens if tk.isalnum()] # will only take the alphanumeric words
    if save_2_file:
        with open('tokens.txt', 'w+') as f:
            f.write(' '.join(tokens))
    return tokens
    
    
# turn tokens into sequences of 30 + 1 length each and save them all in a file
def tkToSeq(tokens, l, save_2_file=False):
    sequences = []
    for i in range(l, len(tokens)):
        curr_seq = tokens[i-l:i]
        joined_seq = ' '.join(curr_seq)
        sequences.append(joined_seq)
    if save_2_file:
        with open('sequences.txt', 'w+') as f:
            ordered_seqs = '\n'.join(sequences)
            f.write(ordered_seqs)
    return sequences


# convert words to integers using tokenizer, splits seqs into X,y and do to_categorical on y
def wordToNums(seqs):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(seqs) # create/update the vocabulary based on input (sequence of) texts
    int_seqs = tokenizer.texts_to_sequences(seqs) # converts word to int sequences
    vocab_size = len(tokenizer.word_index) + 1
    seqs = np.array(int_seqs) # converting list of lists to numpy (n,m) - including the 2nd dimension
    X = seqs[:, :-1]
    y = seqs[:, -1]
    y = to_categorical(y, num_classes=vocab_size) # one-hot encoding for the last word in the seq
    each_seq_length = X.shape[1]
    return X, y, vocab_size, each_seq_length, tokenizer


four_grams = []
with open('allsubs_top10_4grams.txt', 'r') as f:
    four_grams = f.readlines()
    
five_grams = []
with open('allsubs_top10_5grams.txt', 'r') as f:
    five_grams = f.readlines()
    

df_S = pd.read_csv('stopsmoking_allsubmissions_clean.tsv', sep='\t')
dfS = df_S.drop_duplicates(subset=['selftext'])

df_C = pd.read_csv('stopsmoking_allcomments_clean.tsv', sep='\t')
dfC = df_C.drop_duplicates(subset=['body'])

beforedf_C =dfC.loc[df_C['created_utc'] < datetime.datetime(2020, 3, 10, 0, 0).timestamp()]
afterdf_C =dfC.loc[df_C['created_utc'] > datetime.datetime(2020, 3, 11, 0, 0).timestamp()]

beforedf_S =dfS.loc[df_S['created_utc'] < datetime.datetime(2020, 3, 10, 0, 0).timestamp()]
afterdf_S =dfS.loc[df_S['created_utc'] > datetime.datetime(2020, 3, 11, 0, 0).timestamp()]


tokens = cleanText(beforedf_S, 'selftext') # 'body' for comments
seqs = tkToSeq(tokens, 51, True)
X, y, v_size, seq_length, tokenizer = wordToNums(seqs)



# building RNN model here
model = Sequential()
model.add(Embedding(v_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True)) # return sequence to pass the hidden state to the next layer
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(v_size, activation='softmax')) # softmax to give the probabilities of each word to follow the sequence
print(model.summary())

# next compile and fit the model
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.fit(X, y, batch_size=128, epochs=100, verbose=2)

model.save('before_sub_model.h5') # saving the model

with open('before_sub_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)










