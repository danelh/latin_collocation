import math
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

class DefaultCollectionMethod():
    def __init__(self, t_tsh=4.0, freq_tsh=0.1):
        self.g_counter = defaultdict(lambda: defaultdict(int))
        self.lemma_counter = defaultdict(int)
        self.total_groups = 0
        self.t_tsh = t_tsh
        self.freq_tsh = freq_tsh

    def parse_lemmas_in_group(self, grp):
        lemmas = {x[1] for x in grp}
        for l in lemmas:
            self.lemma_counter[l] += 1
            for l2 in lemmas-{l}:
                self.g_counter[l][l2] += 1
        self.total_groups += 1

    def find(self):

        lemmas_freq_in_group = {l: float(self.lemma_counter[l]) / self.total_groups for l in self.lemma_counter}
        for l, l_group in self.g_counter.items():
            for paired_l, paried_l_count in l_group.items():
                if lemmas_freq_in_group[l] < self.freq_tsh and lemmas_freq_in_group[paired_l] < self.freq_tsh:
                    t = self.analyze_pair(l, paired_l, lemmas_freq_in_group)
                    if t > self.t_tsh:
                        print(l, paired_l, t)

    def analyze_pair(self, l1, l2, lemmas_freq_in_group):
        p1 = lemmas_freq_in_group[l1]
        p2 = lemmas_freq_in_group[l2]
        m = p1 * p2
        x = float(self.g_counter[l1][l2]) / self.total_groups
        s2 = x * (1 - x)
        t = (x - m) / math.sqrt(s2 / self.total_groups)
        return t

class WordDistanceCollectionMethod():
    def __init__(self, word_distance, t_tsh=4.0, freq_tsh=0.1):
        self.word_distance = word_distance
        self.default_collection = DefaultCollectionMethod(t_tsh=t_tsh, freq_tsh=freq_tsh)

    def parse_lemmas_in_group(self, grp):
        # we want groups of word_distance*2 + 1 .(from each side)
        # every group only the middle word count (unless in the end or beginning)
        # for index (lemma) which slice we have to take:
        # 0: (0, 0 + word_distance)
        # 1: (0, 1 + wd)
        # n: (max(0, n-wd), min(n+wd, length-1)) [length-1 because this is the last element]
        # note that the slice "included" in both sides
        # lemmas_list = [x[1] for x in grp]
        grp_fixed_list = list(grp)
        for i, l in enumerate(grp):
            start = max(0, i-self.word_distance)
            end = min(len(grp)-1, i + self.word_distance)
            _slice = grp_fixed_list[start:end+1]
            self.default_collection.parse_lemmas_in_group(_slice)

    def find(self):
        return self.default_collection.find()

class CollocationCollector():
    def __init__(self, lemmatizer, tokenizer, collection_method):
        self.lemmatizer = lemmatizer
        self.tokenizer = tokenizer
        self.collection_method = collection_method
        # self.g_counter = defaultdict(lambda: defaultdict(int))
        # self.lemma_counter = defaultdict(int)
        # self.total_groups = 0

    def parse(self, sentences):
        for s in sentences:
            self.parse_sentence(s)

    def parse_sentence(self, sen):
        # tokens = self.tokenizer.tokenize(sen)
        # we don't need tokenizer at all (for poetry):
        tokens = [sen]
        for t in tokens:
            self.parse_token(t)

    def parse_token(self, token):
        # self.parse_lemmas_in_group()
        self.collection_method.parse_lemmas_in_group(self.lemmatize_token(token))

    def find_collocation(self):
        return self.collection_method.find()

    def lemmatize_token(self, token):
        regex = re.compile('[^a-zA-Z]')
        words = [regex.sub('', x) for x in token.split()]
        return self.lemmatizer.lemmatize(words)

    # def parse_lemmas_in_group(self, grp):
    #     lemmas = {x[1] for x in grp}
    #     for l in lemmas:
    #         self.lemma_counter[l] += 1
    #         for l2 in lemmas-{l}:
    #             self.g_counter[l][l2] += 1
    #     self.total_groups += 1

    # def find_collocation(self):
    #     # total_lemmas = sum(self.lemma_counter.values())
    #     tsh = 1 / 30000.0
    #     lemmas_freq_in_group = {l: float(self.lemma_counter[l]) / self.total_groups for l in self.lemma_counter}
    #     for l, l_group in self.g_counter.items():
    #         for paired_l, paried_l_count in l_group.items():
    #             if lemmas_freq_in_group[l] < 0.1 and lemmas_freq_in_group[paired_l] < 0.1:
    #                 t= self.analyze_pair(l, paired_l, lemmas_freq_in_group)
    #                 if t > 4:
    #                     print(l, paired_l, t)
    #
    # def analyze_pair(self, l1, l2, lemmas_freq_in_group):
    #     p1 = lemmas_freq_in_group[l1]
    #     p2 = lemmas_freq_in_group[l2]
    #     m = p1*p2
    #     x = float(self.g_counter[l1][l2]) / self.total_groups
    #     s2 = x * (1-x)
    #     t = (x - m) / math.sqrt(s2/self.total_groups)
    #     return t



tokenizer = LineTokenizer('latin')
lemmatizer = BackoffLatinLemmatizer()

reader = get_corpus_reader(language='latin', corpus_name='latin_text_perseus')
docs = list(reader.docs())
# reader._fileids = ['cicero__on-behalf-of-aulus-caecina__latin.json']
sentences = list(reader.sents())

# to speedup
sentences = sentences[::2]

print (len(sentences))
# collection_method = DefaultCollectionMethod()
# collection_method = WordDistanceCollectionMethod(2, t_tsh=3, freq_tsh=0.01)
collection_method = WordDistanceCollectionMethod(1, t_tsh=2.5, freq_tsh=0.01)
cc = CollocationCollector(lemmatizer, tokenizer, collection_method)
cc.parse(sentences)
cc.find_collocation()
# print (cc.g_counter["capio"])
print ("test")


# very good examples in "stringo" "lacus" "iaceo corpus"