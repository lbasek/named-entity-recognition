import numpy as np

file_glove = open("./glove.6B.100d.txt", encoding="utf-8")

word2Idx = {}
wordEmbeddings = []

for line in file_glove:
    split = line.strip().split(" ")
    word = split[0]
    print(word)
