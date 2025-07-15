#!/usr/bin/env python3

import sys

with open(sys.argv[1]) as file1, \
     open(sys.argv[2]) as file2:
    for line1, line2 in zip(file1, file2):
        line1 = line1.strip()
        if line1:
            lemma = line2.rstrip().split("\t")[-1]
            print(line1, lemma, sep="\t")
        else:
            print()
