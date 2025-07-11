#!/usr/bin/env python3

import sys

with open(sys.argv[1]) as file1, \
     open(sys.argv[2]) as file2:
    for line1, line2 in zip(file1, file2):
        if line1.strip():
            word = line1.strip().split("\t")[0]
            tag_lemma = line2.strip().split("\t")[1:]
            print(word, tag_lemma, sep="\t")
        else:
            print()
