import spacy
nlp = spacy.load('en')

import neuralcoref
neuralcoref.add_to_pipe(nlp)

sentences = ['My sister has a dog. She loves him.', 'Some like to play football, others are fond of basketball.', 'The more a man knows, the more he feels his ignorance.']

for s in sentences:
    print("句子：{}".format(s))
    doc = nlp(s)
    if doc._.has_coref:
        chains = doc._.coref_clusters
        print('共指链：')
        num_chains = len(chains)
        for i in range(num_chains):
            chain = chains[i]
            num = len(chain)
            for j in range(num - 1):
                if j is 0:
                    print("  链{}".format(j))
                    print("    ", end="")
                print("{}-->".format(chain[j]), end="")
            print("{}\n".format(chain[num - 1]), end="")
    else:
        print('没有共指链')