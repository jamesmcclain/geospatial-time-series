#!/usr/bin/env python3

import gensim.downloader as api
model = api.load('glove-twitter-25')

import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
