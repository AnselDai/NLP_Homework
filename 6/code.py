import nltk

grammar = nltk.CFG.fromstring("""
 S -> NP VP
 VP -> VBD NP | VBD NP PP
 PP -> IN NP
 NP -> DT NN | DT NN PP
 DT -> "the" | "a"
 NN -> "boy" | "dog" | "rod"
 VBD -> "saw"
 IN -> "with"
""")
words = nltk.word_tokenize("the boy saw the dog with a rod")
tags = nltk.pos_tag(words)
rd_parser = nltk.RecursiveDescentParser(grammar)
for tree in rd_parser.parse(words):
    print(tree)