import spacy
nlp = spacy.load('en')
from spacy.symbols import VERB, ADJ, nsubj, dobj
from collections import defaultdict, Counter
from nltk.corpus import stopwords
import math
import operator
import numpy as np

class Characterize:
    def __init__(self, term_dict, terms):
        self.term_dict = term_dict
        self.terms = terms
        char_dict = self.run_char()
        attr_dict, term_dict = self.characterize_by_term(char_dict)
        self.get_prominent_features(term_dict)
        self.get_distinct_features(attr_dict)



    def get_deps(self, entities, term):
        agent, patient, mods, poss= [], [], [], []
        for ent in entities:
            if ent.text == term:
                # actions
                if ent.dep == nsubj and ent.head.pos == VERB:
                    if ent.head.text not in stopwords.words():
                        agent.append(ent.head.text)
                if ent.dep == dobj and ent.head.pos == VERB:
                    if ent.head.text not in stopwords.words():
                        patient.append(ent.head.text)
                #poss
                if ent.dep_ == 'poss':
                    poss.append(ent.head.text)

            #adjectives
            if ent.pos == ADJ and ent.head.text == term:
                mods.append(ent.text)
            elif ent.pos == ADJ and ent.head == ent.head:
                mods.append(ent.text)

        return agent, patient, mods, poss

    def run_char(self):
        count = 0
        characterized = defaultdict(dict)
        for term, doc_passages in self.term_dict.items():
            char = {}
            for doc_id, passages in doc_passages.items():
                p_agent, p_pat, p_mod, p_poss = [], [], [], []
                for passage in passages:
                    count += 1
                    doc = nlp(passage)
                    agent, patient, mods, poss = self.get_deps(doc, term)

                    p_agent.extend(agent)
                    p_pat.extend(patient)
                    p_mod.extend(mods)
                    p_poss.extend(poss)

                char[doc_id] = {'agent': p_agent,
                                'patient' : p_pat,
                                'mods': p_mod,
                                'poss': p_poss}

            characterized[term].update(char)
        return characterized


    def characterize_by_term(self, char_dict):
        attr_dict = defaultdict(lambda: defaultdict(dict))
        term_dict = defaultdict(lambda: defaultdict(dict))
        for term, docs in char_dict.items():
            for doc_id, chars in docs.items():
                for w in chars['agent']:
                    attr_dict['agent'][term].setdefault(w, 0)
                    attr_dict['agent'][term][w] += 1
                    term_dict[term]['agent'].setdefault(w, 0)
                    term_dict[term]['agent'][w] += 1

                for w in chars['patient']:
                    attr_dict['patient'][term].setdefault(w, 0)
                    attr_dict['patient'][term][w] += 1
                    term_dict[term]['patient'].setdefault(w, 0)
                    term_dict[term]['patient'][w] += 1

                for w in chars['mods']:
                    attr_dict['mods'][term].setdefault(w, 0)
                    attr_dict['mods'][term][w] += 1
                    term_dict[term]['mods'].setdefault(w, 0)
                    term_dict[term]['mods'][w] += 1

                for w in chars['poss']:
                    attr_dict['poss'][term].setdefault(w, 0)
                    attr_dict['poss'][term][w] += 1
                    term_dict[term]['poss'].setdefault(w, 0)
                    term_dict[term]['poss'][w] += 1

        return attr_dict, term_dict

    def get_prominent_features(self, term_dict):
        for term in term_dict:
            print(term)
            atotal = sum(term_dict[term]['agent'].values())
            pattotal = sum(term_dict[term]['patient'].values())
            mtotal = sum(term_dict[term]['mods'].values())
            ptotal = sum(term_dict[term]['poss'].values())
            if atotal != 0:
                for key in term_dict[term]['agent']:
                    term_dict[term]['agent'][key] /= atotal
            if pattotal != 0:
                for key in term_dict[term]['patient']:
                    term_dict[term]['patient'][key] /= pattotal
            if mtotal != 0:
                for key in term_dict[term]['mods']:
                    term_dict[term]['mods'][key] /= mtotal
            if ptotal != 0:
                for key in term_dict[term]['poss']:
                    term_dict[term]['poss'][key] /= ptotal

            print('agent')
            for word, val in Counter(term_dict[term]['agent']).most_common(20):
                print('{0}\t{1}'.format(word, np.round(val, 3)))
            print('patient')
            for word, val in Counter(term_dict[term]['patient']).most_common(20):
                print('{0}\t{1}'.format(word, np.round(val, 3)))
            print('mods')
            for word, val in Counter(term_dict[term]['mods']).most_common(20):
                print('{0}\t{1}'.format(word, np.round(val, 3)))
            print('poss')
            for word, val in Counter(term_dict[term]['poss']).most_common(20):
                print('{0}\t{1}'.format(word, np.round(val, 3)))

    def get_distinct_features(self, attr_dict):
        for attr in attr_dict:
            assert len(self.terms) == 2
            self.calculate_differences(Counter(attr_dict[attr][self.terms[0]]), Counter(attr_dict[attr][self.terms[1]]), attr)

    def calculate_differences(self, counter1, counter2, category, display=25):

        """ Function that takes two Counter objects as inputs and prints out a ranked list of terms
        more characteristic of the first counter than the second.  Here we'll use log-odds
        with an uninformative prior (from Monroe et al 2008, "Fightin Words", eqn. 22) as our metric.

        """
        vocab = dict(counter1)
        vocab.update(dict(counter2))
        maleSum = sum(counter1.values())
        femaleSum = sum(counter2.values())

        ranks = {}
        alpha = 0.01
        alphaV = len(vocab) * alpha

        for word in vocab:
            log_odds_ratio = math.log(
                (counter1[word] + alpha) / (maleSum + alphaV - counter1[word] - alpha)) - math.log(
                (counter2[word] + alpha) / (femaleSum + alphaV - counter2[word] - alpha))
            variance = 1. / (counter1[word] + alpha) + 1. / (counter2[word] + alpha)

            ranks[word] = log_odds_ratio / math.sqrt(variance)

        sorted_x = sorted(ranks.items(), key=operator.itemgetter(1), reverse=True)

        print(category)
        print("{}".format(self.terms[0]))
        for k, v in sorted_x[:display]:
            print("%.3f\t%s" % (v, k))

        print("\n{}".format(self.terms[1]))
        for k, v in reversed(sorted_x[-display:]):
            print("%.3f\t%s" % (v, k))

