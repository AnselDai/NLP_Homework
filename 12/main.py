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
    print(sentences)
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
    for review in reviews["review"]:
        clean_reviews.append(review_to_wordlist(review, remove_stopwords=True))
    return clean_reviews

def create_bag_of_centroids(wordlist, word_centroid_map):
    num_centroids = max(list(word_centroid_map.values()))
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    return bag_of_centroids

num_features = 300
min_word_count = 40
num_workers = 10
context = 10
downsampling = 1e-3

if __name__ == '__main__':
    print("Begin to read training data")
    train = pd.read_csv('./data/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
    unlabeled_train = pd.read_csv('./data/unlabeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("./data/testData.tsv", header=0, delimiter="\t", quoting=3)
    print("Complete to read training data")
    print('Read {} labeled train reviews, {} unlabeled reviews, {} test reviews'.format(train['review'].size,
                                                                                        unlabeled_train['review'].size,
                                                                                        test['review'].size))
    print("------------------------")
    model_name = "300features_40minwords_10context"
    if os.path.exists(model_name):
        model = Word2Vec.load(model_name)
    else:
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = []
        print('Parsing sentences from training set')
        for review in train["review"]:
            sentences += review_to_sentences(review, tokenizer)
        print('Parsing sentences from unlabeled set')
        for review in unlabeled_train['review']:
            sentences += review_to_sentences(review, tokenizer)
        print("Training model ...")
        model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count, window=context, sample=downsampling)
        model.init_sims(replace=True)
        model.save(model_name)
    print("Test model...")
    print(model.doesnt_match("france england germany berlin".split()))
    print(model.doesnt_match("paris berlin london austria".split()))
    print(model.most_similar("man"))
    print(model.most_similar("queen"))
    print(model.most_similar("awful"))
    print("-----------------------------------")

    start = time.time()

    word_vectors = model.wv.syn0
    num_clusters = int(word_vectors.shape[0] / 5)

    # init kmeans and use it to extract centroids
    kmeans_clustering = KMeans(n_clusters=num_clusters)
    idx = kmeans_clustering.fit_predict(word_vectors)

    end = time.time()
    elapsed = end - start
    print("Time taken for K Means clustering: {} seconds".format(elapsed))

    word_centroid_map = dict(zip(model.wv.index2word, idx))
    for cluster in range(0, 10):
        print("Cluster {}".format(cluster))
        words = []
        for i in range(0, len(word_centroid_map.values())):
            if (list(word_centroid_map.values())[i] == cluster):
                words.append(list(word_centroid_map.keys())[i])
        print(words)

    # clean_train_reviews = getCleanReviews(train)
    # trainDataVecs = getAvgFeatureVec(clean_train_reviews, model, num_features)
    # clean_test_reviews = getCleanReviews(test)
    # testDataVecs = getAvgFeatureVec(clean_test_reviews, model, num_features)
    #
    # forest = RandomForestClassifier(n_estimators=100)
    # forest = forest.fit(trainDataVecs, train["sentiment"])
    # result = forest.predict(testDataVecs)
    #
    # output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    # output.to_csv("Word2Vec_AverageVectors.csv", index=False, quoting=3)

    # num_reviews = train['review'].size
    # clean_train_reviews = []
    # for i in range(0, num_reviews):
    #     if ((i+1) % 1000) is 0:
    #         print("Training review {} of {}".format(i+1, num_reviews))
    #     clean_train_reviews.append(review_to_words(train["review"][i]))
    # vectorizer = CountVectorizer(analyzer="word",  \
    #                             tokenizer=None,   \
    #                             preprocessor=None,\
    #                             max_features=5000)
    # train_data_features = vectorizer.fit_transform(clean_train_reviews)
    # train_data_features = train_data_features.toarray()
    # # vocab = vectorizer.get_feature_names()
    # # print(vocab)
    # # dist = np.sum(train_data_features, axis=0)
    # # for tag, count in zip(vocab, dist):
    # #     print("{}, {}".format(tag, count))
    # print("Start Training")
    # model = Path("./models/Forest.m")
    # if model.is_file():
    #     forest = joblib.load("./models/Forest.m")
    # else:
    #     forest = RandomForestClassifier(n_estimators=100)
    #     forest = forest.fit(train_data_features, train["sentiment"])
    #     f = open("./models/Forest.m", "w")
    #     f.close()
    #     joblib.dump(forest, "./models/Forest.m")
    # print("Finish Training")
    # print("------------------------")
    # print("Begin to read testing data")
    # test = pd.read_csv("./data/testData.tsv", header=0, delimiter="\t", quoting=3)
    # print("Complete to read testing data")
    # print("------------------------")
    # num_reviews = len(test["review"])
    # clean_test_reviews = []
    # for i in range(0, num_reviews):
    #     if (i+1)%1000 == 0:
    #         print("Testing review {} of {}".format(i+1, num_reviews))
    #     clean_test_reviews.append(review_to_words(test["review"][i]))
    # test_data_features = vectorizer.transform(clean_test_reviews)
    # test_data_features = test_data_features.toarray()
    # print("Start predicting")
    # result = forest.predict(test_data_features)
    # print("Finish predicting")
    # print("------------------------")
    # output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    # output.to_csv("./Bag_of_Words_model.csv", index=False, quoting=3)