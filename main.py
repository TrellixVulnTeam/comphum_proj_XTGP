import pandas as pd
from collections import Counter, defaultdict
from preprocess.Process import PreProcessor
from extactor.Extractor import Extractor
from characterize.Characterize import Characterize
from supervised.Model import Model
import numpy as np

#preprocess
f = open('data/docs.txt', 'r')
meta_df = pd.read_csv('data/metadata_sheet.csv')
preproc =  PreProcessor(f, meta_df)
meta = preproc.meta
data = preproc.processed

#extract relevant passages
countries = ['China', 'India', 'Japan', 'England', 'France', 'Italy', 'America']
asian_countries = ['China', 'India', 'Japan']
western_countries = ['England', 'France', 'Italy', 'America']

terms = ['China', 'England']
ext = Extractor(terms, data)
term_passages = ext.term_chunks

#basic statistics:
passage_count = 0
words_in_passage = []
term_count = defaultdict(lambda: defaultdict(dict))
for term, doc_passages in term_passages.items():
    for doc_id, passages in doc_passages.items():
        term_count[term].setdefault('passages', 0)
        term_count[term]['passages'] += len(passages)
        term_count[term].setdefault('words', 0)
        term_count[term]['words'] += sum([len(p.split()) for p in passages])


for term in term_passages:
    print(term)
    print('number of passages', term_count[term]['passages'])
    print('average words', term_count[term]['words']/term_count[term]['passages'])


# characterize passages
char = Characterize(term_passages, terms)
print()

