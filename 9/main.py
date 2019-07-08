import pandas as pd
from bs4 import BeautifulSoup
import warnings
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from pathlib import Path
import os

warnings.filterwarnings('ignore')

def review_to_words(raw_review):
    # 1 .remove HTML
    review_text = BeautifulSoup(raw_review).get_text()
    # 2. remove non-letters
        # search for not a-zA-Z
        # replace the words with " "
        # the text for search is example1.get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    # convert to lower case and split
    words = letters_only.lower().split()
    # 4. search for stopwords
    stops = set(stopwords.words("english"))
    # 5. remove stopwords
    meaningful_words = [w for w in words if not w in stops]
    # 6. join the words into 1 string seperated by space
    return (" ".join(meaningful_words))

if __name__ == '__main__':
    print("Begin to read training data")
    train = pd.read_csv('./data/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
    print("Complete to read training data")
    print("------------------------")
    num_reviews = train['review'].size
    clean_train_reviews = []
    for i in range(0, num_reviews):
        if ((i+1) % 1000) is 0:
            print("Review {} of {}".format(i+1, num_reviews))
        clean_train_reviews.append(review_to_words(train["review"][i]))
    vectorizer = CountVectorizer(analyzer="word",  \
                                tokenizer=None,   \
                                preprocessor=None,\
                                max_features=5000)
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    train_data_features = train_data_features.toarray()
    # vocab = vectorizer.get_feature_names()
    # print(vocab)
    # dist = np.sum(train_data_features, axis=0)
    # for tag, count in zip(vocab, dist):
    #     print("{}, {}".format(tag, count))
    print("Start Training")
    model = Path("./models/Forest.m")
    if model.is_file():
        forest = joblib.load("./models/Forest.m")
    else:
        forest = RandomForestClassifier(n_estimators=100)
        forest = forest.fit(train_data_features, train["sentiment"])
        f = open("./models/Forest.m", "w")
        f.close()
        joblib.dump(forest, "./models/Forest.m")
    print("Finish Training")
    print("------------------------")
    print("Begin to read testing data")
    test = pd.read_csv("./data/testData.tsv", header=0, delimiter="\t", quoting=3)
    print("Complete to read testing data")
    print("------------------------")
    num_reviews = len(test["review"])
    clean_test_reviews = []
    for i in range(0, num_reviews):
        if (i+1)%1000 == 0:
            print("Testing review {} of {}".format(i+1, num_reviews))
        clean_test_reviews.append(review_to_words(test["review"][i]))
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()
    print("Start predicting")
    result = forest.predict(test_data_features)
    print("Finish predicting")
    print("------------------------")
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("./Bag_of_Words_model.csv", index=False, quoting=3)