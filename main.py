from collections import defaultdict

from cltk.corpus.readers import get_corpus_reader
from cltk.lemmatize.latin.backoff import BackoffLatinLemmatizer
from cltk.tokenize.line import LineTokenizer

import re

def _import_corpus():
    from cltk.corpus.utils.importer import CorpusImporter
    corpus_importer = CorpusImporter('latin')
    for x in corpus_importer.list_corpora:
        print (x)
        try:
            corpus_importer.import_corpus(x)
        except Exception as e:
            # probably because expecting local
            print (e)



def parse_sentence(sen, lemmatizer, tokenizer):
    tokens = tokenizer.tokenize(sen)
    for t in tokens:
        words = t.split()
        res = lemmatizer.lemmatize(tokens)
        print(res)


class CollocationCollector():
    def __init__(self, lemmatizer, tokenizer):
        self.lemmatizer = lemmatizer
        self.tokenizer = tokenizer
        self.g_counter = defaultdict(lambda: defaultdict(int))
        self.lemma_counter = defaultdict(int)
        self.total_groups = 0

    def parse(self, sentences):
        for s in sentences:
            self.parse_sentence(s)

    def parse_sentence(self, sen):
        tokens = self.tokenizer.tokenize(sen)
        for t in tokens:
            self.parse_token(t)

    def parse_token(self, token):
        self.parse_lemmas_in_group(self.lemmatize_token(token))

    def lemmatize_token(self, token):
        regex = re.compile('[^a-zA-Z]')
        words = [regex.sub('', x) for x in token.split()]
        return self.lemmatizer.lemmatize(words)

    def parse_lemmas_in_group(self, grp):
        lemmas = {x[1] for x in grp}
        for l in lemmas:
            self.lemma_counter[l] += 1
            for l2 in lemmas-{l}:
                self.g_counter[l][l2] += 1
        self.total_groups += 1

    def find_collocation(self):
        # total_lemmas = sum(self.lemma_counter.values())
        tsh = 1 / 30000.0
        lemmas_freq_in_group = {l: float(self.lemma_counter[l]) / self.total_groups for l in self.lemma_counter}
        for l, l_group in self.g_counter.items():
            if lemmas_freq_in_group[l] < tsh:
                continue
            for paired_l, paried_l_count in l_group.items():
                if lemmas_freq_in_group[paired_l] < tsh:
                    continue
                expected = lemmas_freq_in_group[l] * lemmas_freq_in_group[paired_l] * self.total_groups
                if paried_l_count > 100*expected:
                    if l[0].isupper() or paired_l[0].isupper():
                        continue
                    print (l, paired_l)


tokenizer = LineTokenizer('latin')
lemmatizer = BackoffLatinLemmatizer()

reader = get_corpus_reader(language='latin', corpus_name='latin_text_perseus')
docs = list(reader.docs())
# reader._fileids = ['cicero__on-behalf-of-aulus-caecina__latin.json']
sentences = list(reader.sents())
print (len(sentences))
cc = CollocationCollector(lemmatizer, tokenizer)
cc.parse(sentences)
cc.find_collocation()
# print (cc.g_counter["capio"])
print ("test")


