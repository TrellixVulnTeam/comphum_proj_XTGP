import spacy
nlp = spacy.load('en')
from spacy.symbols import VERB, ADJ, NOUN, AUX, nsubj, dobj, poss
from collections import defaultdict

# Add neural coref to SpaCy's pipe
import neuralcoref
neuralcoref.add_to_pipe(nlp)

# You're done. You can now use NeuralCoref as you usually manipulate a SpaCy document annotations.
class Characterize:
    def __init__(self, term_dict):
        self.term_dict = term_dict
        self.run_char()

    def get_corefs(self, doc, term):
        corefs = []
        token = self.get_tok(doc, term)
        if token != None:
            if doc._.has_coref:
                if token._.in_coref:
                    corefs = token._.coref_clusters
        return corefs

    def get_tok(self, doc, token):
        for ent in doc:
            if ent.text == token:
                return ent
        return None

    def get_mentions(self, corefs, ner):
        entities = []
        for ref in corefs:
            for mention in ref.mentions:
                entities.extend([e for e in ner if e.text == str(mention)])
        entities = set(entities)
        return entities

    def get_deps(self, ner, corefs):
        agent, patient, mods, poss= set(), set(), set(), set()
        entities = self.get_mentions(corefs, ner)

        for ent in entities:
            # actions
            if ent.dep == nsubj and ent.head.pos == VERB:
                agent.add(ent.head.text)
            if ent.dep == dobj and ent.head.pos == VERB:
                patient.add(ent.head.text)
            #poss
            if ent.dep == poss:
                poss.add(ent.head.text)
            #adjectives
            for entity in ner:
                if entity.pos == ADJ and entity.head == ent:
                    mods.add(entity.text)
                elif entity.pos == ADJ and entity.head == ent.head:
                    mods.add(entity.text)

        return agent, patient, mods, poss

    def run_char(self):
        self.characterized = defaultdict(dict)
        for term, doc_passages in self.term_dict.items():
            char = {}
            for doc_id, passages in doc_passages.items():
                p_agent, p_pat, p_mod, p_poss = set(), set(), set(), set()
                for passage in passages:
                    doc = nlp(passage)
                    corefs = self.get_corefs(doc, term)
                    agent, patient, mods, poss = self.get_deps(doc, corefs)

                    p_agent.update(agent)
                    p_pat.update(patient)
                    p_mod.update(mods)
                    p_poss.update(poss)

                char[doc_id] = {'agent': p_agent,
                                'patient' : p_pat,
                                'mods': p_mod,
                                'poss': p_poss}

            self.characterized[term].update(char)

