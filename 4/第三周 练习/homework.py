import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.text import Text
import string
import re

def filter_punctuation(words):
    new_words = []
    illegal_char = string.punctuation + '【·！...（）—：“”？《》、；】'+'\',;.``'
    pattern = re.compile('[%s]]'% re.escape(illegal_char))
    for word in words:
        new_word = pattern.sub(u'', word)
        if not new_word == u'':
            new_words.append(new_word)
    return new_words

with open("./data/text_en.txt", 'r', encoding='UTF-8') as f:
    texts = f.readlines()
words = []
# 分词
for text in texts:
    word = nltk.word_tokenize(text)
    for w in word:
        words.append(w)
print('分词')
print(words)
# 提取词干
word_stem = []
stemmerporte = PorterStemmer()
for word in words:
    stem = stemmerporte.stem(word)
    word_stem.append(stem)
print('提取词干')
print(word_stem)
# 去停用词
stops = set(stopwords.words('english'))
words_no_stop = [word for word in words if word.lower() not in stops]
print('去停用词')
print(words_no_stop)
# 标点符号过滤
words_no_punc = filter_punctuation(words_no_stop)
print('去掉标点符号')
print(words_no_punc)
# 低频词过滤
threshold = 20
words_high_freq = []
fdist = FreqDist(words_no_punc)
for word in words_no_punc:
    freq = fdist[word]
    if freq > 20:
        words_high_freq.append(word)
print('过滤低频词')
print(words_high_freq)
# 绘制离散图，查看指定单词的分布位置
text = Text(words)
pos_words = ['Elizabeth', 'Darcy', 'Wickham', 'Bingley', 'Jane']
print('显示指定单词的位置图')
Text.dispersion_plot(text, pos_words)
# 对前20个有意义的高频词，绘制频率分布图
fdist = FreqDist(words_high_freq)
count = 0
words_selected = []
for sample in fdist:
    number = fdist[sample]
    # print('{0} 出现的次数为：{1}'.format(sample, number))
    for i in range(number):
        words_selected.append(sample)
    count += 1
    if count == 20:
        break
print('前20个有意义的高频词')
# print('{} 最大，为：{}'.format(fdist.max(), fdist[fdist.max()]))
print(words_selected)
fdist_freq = FreqDist(words_selected)
print('绘制频率分布图')
fdist_freq.plot()
