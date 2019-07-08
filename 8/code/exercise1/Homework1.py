from nltk.corpus import wordnet as wn

def get_same_meaning_words(word):
    items = wn.synsets(word) # 获取word的意义集
    res = {}
    res['word'] = word
    res['same_words'] = []
    for item in items:
        names = item.lemma_names()
        names_labels = item.lemmas() # 获取该意义的同义词词条集
        num = len(names)
        for i in range(num):
            single = {}
            single['word'] = names[i]
            m = names_labels[i].synset()
            single['definition'] = m.definition()
            examples = m.examples()
            if len(examples) is 0:
                examples = ['无例子']
            single['examples'] = examples
            res['same_words'].append(single)
    return res

def print_res(res):
    print("{}的同义词及其定义和例子：".format(res['word']))
    for s in res['same_words']:
        print('    {}\n      definition: {}'.format(s['word'], s['definition']))
        first = True
        for e in s['examples']:
            if first:
                print('      examples: {}'.format(e))
                first = False
            else:
                print('                {}'.format(e))

words = ['dog', 'apple', 'fly']
for word in words:
    res = get_same_meaning_words(word)
    print_res(res)