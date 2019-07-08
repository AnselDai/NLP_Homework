import re

key = 'sysu@hotmail.edu.cn'
pattern = r'@[A-Za-z0-9]+?\.'

p1 = re.compile(pattern)
print(p1.findall(key))