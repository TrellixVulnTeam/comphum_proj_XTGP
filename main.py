import pandas as pd
from preprocess.Process import PreProcessor
from extactor.Extractor import Extractor
from characterize.Characterize import Characterize
from supervised.Model import Model

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
compare_countries = ['China', 'England']

ext = Extractor(compare_countries, data)
term_passages = ext.term_chunks

#characterize passages
# char = Characterize(term_passages)
# char_dict = char.characterized
# print()

#build supervised model
mdl = Model(term_passages)
mdl.run_model()
print()

