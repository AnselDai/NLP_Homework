import re

key = r'http://www.poshoaiu.com and https://iusdhbfw.com'
p1 = r'https?://'

pattern = re.compile(p1)
print(pattern.findall(key))