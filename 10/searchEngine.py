from correction import *
from initEngine import *

init()

word = input("输入：")
c_word = correction(word)
print(c_word)
if c_word != word:
    print("你可能是想输入：{}".format(c_word))
    choice = input("是否需要更正？输入0确认修改，输入1保持不变\n")
    if choice == '0':
        word = c_word
print(word)
search(word)