import nltk

def load_training_dataset():
    name = 'training_set.txt'
    labels = []
    sentences = []
    with open(name, 'r', encoding='UTF-8') as f:
        while True:
            data = f.readline()
            if not data:
                break
            if data[0] is '+':
                labels.append(1)
            elif data[0] is '-':
                labels.append(0)
            if data[len(data) - 1] == '\n':
                sentences.append(data[2:len(data) - 1])
            else:
                sentences.append(data[2:len(data)])
    words = []
    for sentence in sentences:
        temp = []
        word = nltk.word_tokenize(sentence)
        for w in word:
            temp.append(w)
        words.append(temp)
    V = 0
    for w in words:
        V += len(w)
    pos_dict = dict()
    neg_dict = dict()
    word_dict = [neg_dict, pos_dict]
    for index in range(len(words)):
        for w in words[index]:
            if word_dict[labels[index]].get(w) is None:
                word_dict[labels[index]][w] = 0
            word_dict[labels[index]][w] += 1
    return word_dict, labels, V

def load_testing_data():
    name = 'testing_set.txt'
    sentences = []
    with open(name, 'r', encoding='UTF-8') as f:
        while True:
            data = f.readline()
            if not data:
                break
            sentences.append(data[0:len(data) - 1])
    words = []
    for s in sentences:
        temp = []
        word = nltk.word_tokenize(s)
        for w in word:
            temp.append(w)
        words.append(temp)
    return words

def predict(sentences, word_dict, V, prior, likehoods):
    C = []
    for c in range(len(word_dict)):
        C.append(0)
        for key in word_dict[c]:
            C[c] += word_dict[c][key]
    res = []
    for s in sentences:
        max_p = -1
        max_c = -1
        for c in range(len(word_dict)):
            p = 1.0
            for w in s:
                p *= caculate_NB(word_dict, w, c, V)
            p *= prior[c]
            if p > max_p:
                max_p = p
                max_c = c
        res.append(max_c)
    return res


def caculate_prior(word_dict, V):
    prior = []
    for c in range(len(word_dict)):
        C = 0
        for key in word_dict[c]:
            C += word_dict[c][key]
        prior.append(C/V)
    return prior


def caculate_NB(word_dict, w, c, V):
    C0 = 0
    for key in word_dict[c]:
        C0 += word_dict[c][key]
    if word_dict[c].get(w) == None:
        C1 = 0
    else:
        C1 = word_dict[c][w]
    p = (C1 + 1)/(C0 + V)
    return p

t_word_dict, t_labels, V = load_training_dataset()
prior = caculate_prior(t_word_dict, V)
likehood = [dict(), dict()]
for c in range(2):
    for key in t_word_dict[c]:
        p = caculate_NB(t_word_dict, key, c, V)
        likehood[c][key] = p
print("先验概率：\n    positive: {}\n    negative: {}".format(prior[0], prior[1]))
print("每个单词的似然：")
for c in range(2):
    if c == 0:
        print("    positive:")
    else:
        print("    negative:")
    for key in likehood[c]:
        print("         {}: {}".format(key, likehood[c][key]))
test_data = load_testing_data()
test_res = predict(test_data, t_word_dict, V, prior, likehood)
print("测试结果：")
for i in range(len(test_data)):
    if test_res[i] == 0:
        res = '-'
    else:
        res = '+'
    print("{}: {}".format(test_data[i], res))

