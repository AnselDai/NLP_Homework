import string
import re
import nltk

def read_dict():
    words_dict = dict()
    with open('./corpus.dict.txt', 'r', encoding='UTF-8') as f:
        while True:
            word = f.readline()
            if word[len(word)-1:len(word)] == '\n':
                word = word[0:len(word)-1]
            if not word:
                break
            words_dict[hash(word)] = word
    return words_dict

def write_res(words, name):
    with open(name, 'w', encoding='UTF-8') as f:
        f.write(words)

def read_sentences():
    with open('./corpus.sentence.txt', 'r', encoding='UTF-8') as f:
        sentences = f.read()
    return sentences

def check_dict(words_dict, word):
    if words_dict.get(hash(word)) == None:
        return False
    else:
        return True

def FMM(sentences, words_dict):
    s = sentences
    words = ''
    words_token = []
    while not s == '':
        e = len(s)
        while e > 0 and e <= len(s):
            if (check_dict(words_dict, s[0:e])) or (e == 1):
                words_token.append(s[0:e])
                words += s[0:e]
                words += '/'
                s = s[e:len(s)]
                break
            else:
                e -= 1
    return words, words_token

def BMM(sentences, words_dict):
    s = sentences
    words = ''
    words_token = []
    while not s == '':
        b = 0
        while b >= 0 and b < len(s):
            if (check_dict(words_dict, s[b:len(s)])) or (b == len(s)-1):
                words_token.append(s[b:len(s)])
                words += s[b:len(s)]
                words += '/'
                s = s[0:b]
                break
            else:
                b += 1
    return words, words_token

def Bi_direction_MM(sentences, words_dict):
    FMM_res = FMM(sentences, words_dict)
    BMM_res = BMM(sentences, words_dict)
    if FMM_res[0] == BMM_res[0]:
        return FMM_res
    else:
        f_len = len(FMM_res[1])
        b_len = len(BMM_res[1])
        if f_len < b_len:
            return FMM_res
        else:
            return BMM_res

def filter_punctuation(words):
    new_words = ''
    illegal_char = string.punctuation + '【！…（）—：“”？《》，:。、】'
    pattern = re.compile('[%s]' % re.escape(illegal_char))
    for word in words:
        new_word = pattern.sub(u'', word)
        if not new_word == u'':
            new_words += new_word
    return new_words

def property_note(words_token):
    return nltk.pos_tag(words_token)

# 读取字典
words_dict = read_dict()
# 读取文本并处理标点符号
sentences =  filter_punctuation(read_sentences())
# 最大正向匹配分词
words, words_token = FMM(sentences, words_dict)
words, words_token = Bi_direction_MM(sentences, words_dict)
print(words)
# 保存结果
# write_res(words)
# print("分词结果保存在了 ./corpus.out.txt 文件中")
# 词性标注
tags = property_note(words_token)
print(tags)
