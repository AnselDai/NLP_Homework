import pandas as pd
from bs4 import BeautifulSoup
import warnings
import re
from nltk.corpus import stopwords
import nltk.data
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from pathlib import Path
import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import time

warnings.filterwarnings('ignore')

def review_to_wordlist(raw_review, remove_stopwords=False):
    # 1 .remove HTML
    review_text = BeautifulSoup(raw_review).get_text()
    # 2. remove non-letters
        # search for not a-zA-Z
        # replace the words with " "
        # the text for search is example1.get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    # convert to lower case and split
    words = letters_only.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    # 6. join the words into 1 string seperated by space
    return (words)

def review_to_sentences(review, tokenizer, remove_stopwords=False):
    # 1. use the NLTK tokenizers to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    # 2. loop over each sentences
    sentences = []
    for raw_sentence in raw_sentences:
        # if empty, skip
        if len(raw_sentence) > 0:
            # otherwise call review_to_wordlist to get list of wordlist
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    return sentences

def makeFeatureVec(words, model, num_feature):
    # average all word vectors in a paragraph
    featureVec = np.zeros((num_feature, ), dtype="float32")
    nwords = 0
    # index2word: model, contains names of words in the model's vocabulary
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            nwords += 1
            featureVec = np.add(featureVec, model[word])
    featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVec(reviews, model, num_features):
    counter = 0
    reviewFeatureVec = np.zeros((len(reviews), num_features), dtype="float32")
    for review in reviews:
        if counter % 1000 == 0:
            print("Review {} of {}".format(counter, len(reviews)))
        reviewFeatureVec[counter] = makeFeatureVec(review, model, num_features)
        counter += 1
    return reviewFeatureVec

def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["text"]:
        clean_reviews.append(review_to_wordlist(review, remove_stopwords=True))
    return clean_reviews

num_features = 300
min_word_count = 40
num_workers = 10
context = 10
downsampling = 1e-3

if __name__ == '__main__':
    print("Begin to read training data")
    train = pd.read_csv('./data/train.tsv', header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("./data/test.tsv", header=0, delimiter="\t", quoting=3)
    print("Complete to read training data")
    print('Read {} labeled train reviews, {} test reviews'.format(train['text'].size,
                                                                  test['text'].size))
    print("------------------------")
    model_name = "300features_40minwords_10context_hm1"
    if os.path.exists(model_name):
        model = Word2Vec.load(model_name)
    else:
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = []
        print('Parsing sentences from training set')
        for review in train["text"]:
            sentences += review_to_sentences(review, tokenizer)
        print('Parsing sentences from unlabeled set')
        print("Training model ...")
        model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count, window=context, sample=downsampling)
        model.init_sims(replace=True)
        model.save(model_name)

    clean_train_reviews = getCleanReviews(train)
    trainDataVecs = getAvgFeatureVec(clean_train_reviews, model, num_features)
    clean_test_reviews = getCleanReviews(test)
    testDataVecs = getAvgFeatureVec(clean_test_reviews, model, num_features)

    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(trainDataVecs, train["polarity"])
    result = forest.predict(testDataVecs)
    print(result)

    output = pd.DataFrame(data={"id": test["line_num"], "polarity": result})
    output.to_csv("Word2Vec_AverageVectors_hm.csv", index=False, quoting=3)