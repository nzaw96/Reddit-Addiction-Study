#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:06:43 2022

@author: Nay Zaw Aung Win
"""
# Code adapted from Professor Manikonda's original code

import pandas as pd
import datetime
# from psaw import PushshiftAPI
import nltk

import re
# from bertopic import BERTopic
from nltk.tokenize import word_tokenize
import operator
from nltk import ngrams
from nltk.corpus import stopwords
nltk.download('punkt')


# cleaning the text data
 

def getTopNGrams(grams_lst, n): # grams_lst is a list of lists with grams and frequency in each
    topN = []
    ignore_wrds = ['https', 'www', 'reddit', 'nlm', 'ncbi', 'x200b', 'pubmed',
                   'google-store', 'edu']
    i = 0
    while len(topN) <= n:
        grams = grams_lst[i][0]
        ignoreGram = False
        for ignore_wrd in ignore_wrds:
            if ignore_wrd in grams:    
                ignoreGram = True
                break
        if not ignoreGram:
            topN.append(grams)
        i += 1
    return topN


def getBiTriGrams(df, col, stopwrds):
    bidiction = {}
    tridiction = {}
    allTexts = df[col]
    for text in allTexts:
        candidates = word_tokenize(text)
        tokens = []
        for tk in candidates:
            if tk not in stopwrds:
                tokens.append(tk)
        bigrams = ngrams(tokens, 2)
        trigrams = ngrams(tokens, 3)
        for grams in bigrams:
            joined_grams = '-'.join(grams)
            if joined_grams in bidiction:
                bidiction[joined_grams] += 1
            else:
                bidiction[joined_grams] = 1
        for grams in trigrams:
            joined_grams = '-'.join(grams)
            if joined_grams in tridiction:
                tridiction[joined_grams] += 1
            else:
                tridiction[joined_grams] = 1
    sorted_bi = sorted(bidiction.items(), key=operator.itemgetter(1), reverse=True)
    sorted_tri = sorted(tridiction.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_bi, sorted_tri


if __name__ == "__main__":
    df_C = pd.read_csv('stopsmoking_allcomments_LIWC.csv')
    df_S = pd.read_csv('stopsmoking_allsubmissions_LIWC.csv')

    stops = list(stopwords.words('english'))
    i=0
    while i<len(stops):
        stops[i]=re.sub(r'\W+', '',stops[i].lower())
        i=i+1
        
    beforedf_C =df_C.loc[df_C['created_utc'] < datetime.datetime(2020, 3, 10, 0, 0).timestamp()]
    afterdf_C =df_C.loc[df_C['created_utc'] > datetime.datetime(2020, 3, 11, 0, 0).timestamp()]

    beforedf_S =df_S.loc[df_S['created_utc'] < datetime.datetime(2020, 3, 10, 0, 0).timestamp()]
    afterdf_S =df_S.loc[df_S['created_utc'] > datetime.datetime(2020, 3, 11, 0, 0).timestamp()]

    bi, tri = getBiTriGrams(beforedf_C, 'body', stops)    
    bi20Grams = getTopNGrams(bi, 50)
    tri20Grams = getTopNGrams(tri, 50)

    
    










