from nltk.corpus import wordnet as wn
import numpy as np

words = [['good', 'beautiful'], ['good', 'bad'], ['dog', 'cat']]
for pair in words:
    sims = []
    s1 = wn.synsets(pair[0])
    s2 = wn.synsets(pair[1])
    score = 0.0
    count = 0.0
    for t1 in s1:
        for t2 in s2:
            sim = t1.path_similarity(t2)
            if sim is None:
                sim = 0.0
            sims.append(sim)
    index = np.argmax(np.array(sims))
    print('{}和{}的相似度：{}'.format(pair[0], pair[1], sims[index]))