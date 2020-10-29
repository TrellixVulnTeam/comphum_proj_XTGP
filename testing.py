import spacy
import neuralcoref
nlp = spacy.load('en')
neuralcoref.add_to_pipe(nlp)

doc = nlp(u"China's traditions'")

for d in doc:
    print(d.text, d.pos_, d.dep_, d.head, list(d.children))
