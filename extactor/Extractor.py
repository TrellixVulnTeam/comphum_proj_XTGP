class Extractor:
    def __init__(self, terms, processed):
        self.processed = processed
        self.terms = terms
        self.term_chunks = self.get_term_chunks()

    def get_chunks(self, term):
        term_exlist = self.terms.copy()
        term_exlist.remove(term)
        passages = {}
        for id, data in self.processed.items():
            relevant, ix = [], 0
            while ix < len(data):
                if term in data[ix]:
                    para = ''
                    # identify relevant chunk
                    for line in data[ix - 2: ix + 2]:
                        para += line
                    # add to relevant list iff it has that term and not the other conflicting terms
                    if not any(word in para for word in term_exlist):
                        relevant.append(para)
                        ix += 3
                ix += 1
            passages[id] = relevant
        return passages

    def get_term_chunks(self):
        terms_chunks = {}
        for term in self.terms:
            terms_chunks[term] = self.get_chunks(term)
        return terms_chunks

