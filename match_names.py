#!/usr/bin/python3

import argparse
import codecs
import re

parser = argparse.ArgumentParser(description="""one sent per line""")
parser.add_argument("--file1", help="names file", required=True)
parser.add_argument("--file2", help="tweets file", required=True)

args = parser.parse_args()
file1 = str(args.file1)
file2 = str(args.file2)


infile = codecs.open(file1, 'r', 'utf8', errors='ignore')
names = infile.readlines()

tweetsfile = codecs.open(file2, 'r', 'utf8', errors='ignore')
tweets = tweetsfile.readlines()

#get names
male_names = []
female_names = []
for line in names:
    seperated = line.split(';')
    f_name = seperated[1]
    if seperated[3]:
        m_name = seperated[3]
        male_names.append(m_name)
    female_names.append(f_name)


retrieved_male = []
retrieved_female = []

#match first names
for line in tweets:
    seperated = line.split('\t')
    fullname = seperated[1]
    firstname = seperated[1].split()[0]
    if firstname in  male_names:
       print(firstname)

