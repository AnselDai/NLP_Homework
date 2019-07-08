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
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import time
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def text_to_wordlist(raw_text, remove_stopwords=False):
    # 1 .remove HTML
    text = BeautifulSoup(raw_text).get_text()
    # 2. remove non-letters
        # search for not a-zA-Z
        # replace the words with " "
        # the text for search is example1.get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", text)
    # convert to lower case and split
    words = letters_only.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    # 6. join the words into 1 string seperated by space
    return (words)

def text_to_sentences(text, tokenizer, remove_stopwords=False):
    # 1. use the NLTK tokenizers to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(text.strip())
    # 2. loop over each sentences
    sentences = []
    for raw_sentence in raw_sentences:
        # if empty, skip
        if len(raw_sentence) > 0:
            # otherwise call text_to_wordlist to get list of wordlist
            sentences.append(text_to_wordlist(raw_sentence, remove_stopwords))
    return sentences

def getCleanTexts(texts):
    clean_texts = []
    for text in texts["text"]:
        clean_texts.append(text_to_wordlist(text, remove_stopwords=True))
    return clean_texts

def create_bag_of_centroids(wordlist, word_centroid_map):
    num_centroids = max(list(word_centroid_map.values())) + 1
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    return bag_of_centroids

def transform_result(res):
    score = []
    for r in res:
        if r == "positive":
            score.append(1)
        elif r == "neutral":
            score.append(0)
        elif r == "negative":
            score.append(-1)
    return score

num_features = 2000
min_word_count = 40
num_workers = 10
context = 10
downsampling = 1e-3
train_time = 10

if __name__ == '__main__':
    print("Begin to read training data")
    pre_train = pd.read_csv('./data/train.tsv', header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("./data/test.tsv", header=0, delimiter="\t", quoting=3)
    print("Complete to read training data")
    print('Read {} labeled train texts, {} test texts'.format(pre_train['text'].size,
                                                              test['text'].size))
    print("------------------------")
    train_res = []
    test_centroids_arr = []
    for t in range(train_time):
        train = dict()
        pre_test = dict()
        x_train,x_test,y_train,y_test = train_test_split(pre_train["text"], pre_train["polarity"],test_size=0.3,random_state=0)
        train["text"] = x_train
        train["polarity"] = y_train
        pre_test["text"] = x_test
        pre_test["polarity"] = y_test
        model_name = "300features_40minwords_10context_hm" + str(t)
        if os.path.exists(model_name):
            model = Word2Vec.load(model_name)
        else:
            tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            sentences = []
            print('Parsing sentences from training set')
            for text in train["text"]:
                sentences += text_to_sentences(text, tokenizer)
            print("Training model ...")
            model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count, window=context, sample=downsampling)
            model.init_sims(replace=True)
            model.save(model_name)

        start = time.time()

        word_vectors = model.wv.syn0
        num_clusters = int(word_vectors.shape[0] / 5)

        # init kmeans and use it to extract centroids
        kmeans_clustering = KMeans(n_clusters=num_clusters)
        idx = kmeans_clustering.fit_predict(word_vectors)

        end = time.time()
        elapsed = end - start

        word_centroid_map = dict(zip(model.wv.index2word, idx))
        for cluster in range(0, 10):
            words = []
            for i in range(0, len(word_centroid_map.values())):
                if (list(word_centroid_map.values())[i] == cluster):
                    words.append(list(word_centroid_map.keys())[i])

        clean_train_texts = getCleanTexts(train)
        clean_test_texts = getCleanTexts(pre_test)
        clean_test_texts_res = getCleanTexts(test)

        train_centroids = np.zeros((train["text"].size, num_clusters), dtype="float32")
        counter = 0
        for text in clean_train_texts:
            train_centroids[counter] = create_bag_of_centroids(text, word_centroid_map)
            counter += 1

        test_centroids = np.zeros((pre_test["text"].size, num_clusters), dtype="float32")
        counter = 0
        for text in clean_test_texts:
            test_centroids[counter] = create_bag_of_centroids(text, word_centroid_map)
            counter += 1

        test_centroids_res = np.zeros((test["text"].size, num_clusters), dtype="float32")
        counter = 0
        for text in clean_test_texts_res:
            test_centroids_res[counter] = create_bag_of_centroids(text, word_centroid_map)
            counter += 1
        test_centroids_arr.append(test_centroids_res)

        forest = RandomForestClassifier(n_estimators=100)
        print("Fitting a random forest to labeled training data ...")
        forest = forest.fit(train_centroids, train["polarity"])
        joblib.dump(forest, "Train" + str(t) + ".m")
        res = forest.predict(test_centroids)
        acc = accuracy_score(transform_result(res), transform_result(pre_test["polarity"]))
        train_res.append(acc)
        print("{}/{} 准确率为：{}".format(t, train_time, acc))

    predictor = joblib.load("Train"+str(np.argmax(train_res))+".m")
    result = predictor.predict(test_centroids_arr[np.argmax(train_res)])
    output = pd.DataFrame(data={"id": test["line_num"], "polarity": result})
    output.to_csv("BagOfCentroids_hm.csv", index=False, quoting=3)