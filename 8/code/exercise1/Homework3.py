from nltk.corpus import wordnet as wn

words = ['walk', 'supply', 'hot']

def get_entailments(word):
    res = []
    sets = wn.synsets(word)
    for s in sets:
        ens = s.entailments()
        for e in ens:
            res.append(e)
    return res

def get_antonyms(word):
    sets = wn.synsets(word)
    res = []
    for s in sets:
        items = s.lemmas()
        for item in items:
            ans = item.antonyms()
            for a in ans:
                res.append(a)
    return res

for word in words:
    print("单词：{}".format(word))
    res = get_entailments(word)
    a_res = get_antonyms(word)
    if len(res) is 0:
        print("没有蕴含")
    else:
        print("蕴含关系：\n{}".format(res))
    if len(a_res) is 0:
        print("没有反义词")
    else:
        print("反义词：\n{}".format(a_res))