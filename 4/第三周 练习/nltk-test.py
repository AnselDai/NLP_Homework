# coding:utf-8
'''
环境：python + jupyter notebook
安装 pip install nltk （http://www.nltk.org/install.html）
'''

import nltk
import re 
import string
# 第一次运行(NTLK自带语料库默认路径下载) : 
# nltk.download()


'''测试是否成功下载'''
#from nltk.corpus import brown
# print(brown.words())
# print(len(brown.sents())) # 句子数
# print(len(brown.words())) # 单词数

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.text import Text
import string
import re

def filter_punctuation(words):
    new_words = []
    illegal_char = string.punctuation + '【·！...（）—：“”？《》、；】'
    pattern = re.compile('[%s]]'% re.escape(illegal_char))
    for word in words:
        new_word = pattern.sub(u'', word)
        if not new_word == u'':
            new_words.append(new_word)
    return new_words

'''英文文本处理'''
'''词性标注'''
# text_en = open(u'./data/text_en.txt',encoding='utf-8',errors='ignore').read()
text="Don't hesitate to ask questions. Be positive." 
# 分句1
from nltk.tokenize import sent_tokenize 
print(sent_tokenize(text))

# 分词1
words=nltk.word_tokenize(text)
print(words)

stops = set(stopwords.words('english'))
words = [word for word in words if word.lower() not in stops]
print(words)

words_no_punc = filter_punctuation(words)
print(words_no_punc)

fdist = FreqDist(words_no_punc)
fdist.plot()