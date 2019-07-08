import nltk
from nltk.corpus import stopwords
import string
import re
import jieba

database = {}

def filter_punctuation(words):
    new_words = []
    illegal_char = string.punctuation + '【！…（）—：“”？《》，:。、】\n'
    pattern = re.compile('[%s]' % re.escape(illegal_char))
    for word in words:
        new_word = pattern.sub(u'', word)
        if not new_word == u'':
            new_words.append(new_word)
    return new_words

def init():
    print("开始初始化引擎...")
    for i in range(1, 194):
        filename = "./data/"+str(i)+".txt"
        with open(filename, encoding='UTF-8') as f:
            sentences = f.read()
        j_word = jieba.cut_for_search(sentences)
        word = []
        for w in j_word:
            word.append(w)
        stops = set(stopwords.words('english'))
        word = [w for w in word if w.lower() not in stops]
        word = filter_punctuation(word)
        for w in word:
            if database.get(w) == None:
                database[w] = []
            isIn = False
            for index in database[w]:
                if index == i:
                    isIn = True
                    break
            if not isIn:
                database[w].append(i)
    # for w in database:
    #     print("{}: {}".format(w, database[w]))
    print("初始化结束")

def search(word):
    if database.get(word) == None:
        print("没有找到结果")
    else:
        print(database[word])
