import numpy as np

def editor_distance(str1, str2):
    len1 = len(str1)+1
    len2 = len(str2)+1
    f = np.zeros([len1, len2])
    for i in range(len1):
        f[i][0] = i
    for j in range(len2):
        f[0][j] = j
    for i in range(1, len1):
        for j in range(1, len2):
            if str1[i-1] == str2[j-1]:
                cost = 0
            else:
                cost = 1
            f[i][j] = min(f[i-1][j-1] + cost, f[i][j-1] + 1, f[i-1][j] + 1)
    return f[-1][-1]

str1 = input('input string1: ')
str2 = input('input string2: ')
print(editor_distance(str1, str2))