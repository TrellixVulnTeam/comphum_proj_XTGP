import pandas as pd

class PreProcessor:
    def __init__(self, f, meta_df):
        self.f = f
        self.metadata_df = meta_df
        self.file_to_docs()
        self.processed = self.preprocess()
        self.meta = self.get_meta()

    def file_to_docs(self, num_files=10):
        data = self.f.read()
        doc_split = data.split("____________________________________________________________")
        self.doc_split = doc_split[1:num_files]

    def preprocess(self,):
        ids = []
        processed = {}
        for ix, doc in enumerate(self.doc_split):
            lines = doc.splitlines()
            id = ''
            for ix, line in enumerate(lines):
                id_list = []
                #trim out irrelevant parts of document
                if 'USTC subject classification' in line:
                    processed_chunk = lines[10: ix - 2]
                #get doc_id
                if 'ProQuest document ID' in line:
                    id_list = [s for s in line.split() if s.isdigit()]
                for num in id_list:
                    id += num
            if id != '':
                processed[id] = processed_chunk
                ids.append(id)
        return processed

    def update_meta(self, id):
        meta = {}
        row = self.metadata_df[self.metadata_df['StoreId'] == int(id)]
        attr = ['Title', 'Authors', 'pubdate', 'subjects']
        comp_data = True
        for a in attr:
            val = list(row[a])
            if len(val) != 0:
                meta.update({a: val[0]})
            else:
                comp_data = False
        return meta, comp_data

    def get_meta(self,):
        ids = list(self.processed.keys())
        metadata_dict = {}
        for id in ids:
            meta_dict, comp_data = self.update_meta(id)
            if comp_data:
                metadata_dict[id] = meta_dict
        return metadata_dict

