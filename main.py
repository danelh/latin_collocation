import json
from json import JSONEncoder

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
            pass
            # if x == "phi5":
            #     corpus_importer.import_corpus(x)
        except Exception as e:
            # probably because expecting local
            print (e)



def parse_sentence(sen, lemmatizer, tokenizer):
    tokens = tokenizer.tokenize(sen)
    for t in tokens:
        words = t.split()
        res = lemmatizer.lemmatize(tokens)
        print(res)

class AbstractCollectionMethod():
    def __init__(self):
        self.pairs = None
        self.ref = None

    def find(self):
        self.pairs = self._get_pairs()
        # print("Here:{}".format(self.get_name()))
        # self.ref = self._get_ref()
        # print("Here2:{}".format(self.get_name()))
        return self.pairs

    def _get_pairs(self):
        return {}

    def _get_ref(self):
        return {}

    def get_name(self):
        return "no_name"

class DefaultCollectionMethod(AbstractCollectionMethod):
    def __init__(self, t_tsh=4.0, freq_tsh=0.1):
        super(AbstractCollectionMethod).__init__()
        self.g_counter = defaultdict(lambda: defaultdict(int))
        self.lemma_counter = defaultdict(int)
        self.total_groups = 0
        self.t_tsh = t_tsh
        self.freq_tsh = freq_tsh
        self._ref = defaultdict(set)
        self.grp_list = []
        self.all_pairs = set()

    def parse_lemmas_in_group(self, grp):
        grp_dict = {x[1]: x[0] for x in grp}
        # due to space issue we should keep the entire grp,
        # and later to run over the required paris only
        self.grp_list.append(grp_dict)

        lemmas = grp_dict.keys()
        for l in lemmas:
            self.lemma_counter[l] += 1
            for l2 in lemmas-{l}:
                self.g_counter[l][l2] += 1
                # frozenset to avoid duplicats. in comment (see above why)
                #self._ref[frozenset((l,l2))].add(frozenset((grp_dict[l],grp_dict[l2])))
        self.total_groups += 1

    def _get_ref(self):
        for pair in self.all_pairs:
            [l1, l2] = list(pair)
            for grp in self.grp_list:
                lemmas_set = set(grp.keys())
                if pair <= lemmas_set:
                    self._ref[pair].add(frozenset((grp[l1], grp[l2])))
        return self._ref

    def _get_pairs(self):

        pairs = defaultdict(lambda: defaultdict(int))
        lemmas_freq_in_group = {l: float(self.lemma_counter[l]) / self.total_groups for l in self.lemma_counter}
        for l, l_group in self.g_counter.items():
            for paired_l, paried_l_count in l_group.items():
                if lemmas_freq_in_group[l] < self.freq_tsh and lemmas_freq_in_group[paired_l] < self.freq_tsh:
                    t = self.analyze_pair(l, paired_l, lemmas_freq_in_group)
                    if t > self.t_tsh:
                        # print(l, paired_l, t)
                        pairs[l][paired_l] = t
                        self.all_pairs.add(frozenset((l, paired_l)))
        return pairs

    def analyze_pair(self, l1, l2, lemmas_freq_in_group):
        p1 = lemmas_freq_in_group[l1]
        p2 = lemmas_freq_in_group[l2]
        m = p1 * p2
        x = float(self.g_counter[l1][l2]) / self.total_groups
        s2 = x * (1 - x)
        t = (x - m) / math.sqrt(s2 / self.total_groups)
        return t

class WordDistanceCollectionMethod(AbstractCollectionMethod):
    def __init__(self, word_distance, t_tsh=4.0, freq_tsh=0.1):
        super(AbstractCollectionMethod).__init__()
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

    def _get_pairs(self):
        return self.default_collection._get_pairs()

    def _get_ref(self):
        return self.default_collection._get_ref()

    def get_name(self):
        return "{}{}".format("w", str(self.word_distance))

class CollocationCollector():
    def __init__(self, lemmatizer, tokenizer, collection_methods):
        self.lemmatizer = lemmatizer
        self.tokenizer = tokenizer
        self.collection_methods = collection_methods
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
        for collection_method in self.collection_methods:
            collection_method.parse_lemmas_in_group(self.lemmatize_token(token))

    def find_collocation(self):
        print ("Here0", len(self.collection_methods))
        return [collection_method.find() for collection_method in self.collection_methods]

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

def merge_collection_methods_to_dict(cms):
    res_dict = {}
    for cm in cms:
        # convert to new dict (and avoid also defaultdict if it is)
        res_dict[cm.get_name()] = {k: dict(v) for k, v in cm.pairs.items()}

    return res_dict

def fix_float_for_json(x):

    def pretty_floats(obj):
        if isinstance(obj, float):
            return round(obj, 2)
        elif isinstance(obj, dict):
            return dict((k, pretty_floats(v)) for k, v in obj.items())
        elif isinstance(obj, (list, tuple)):
            return list(map(pretty_floats, obj))
        return obj

    return json.dumps(pretty_floats(x))

def sort_collectors(d, lim=100):
    for x in d:
        d[x] = sort_collector(d[x], lim)

    return d

def sort_collector(d, lim=100):
    for k, dv in d.items():
        l = [(k,v) for k, v in sorted(dv.items(), key=lambda item: item[1], reverse=True)]
        d[k] = l[:lim]
    return d

def save_collectors(cms):
    d = merge_collection_methods_to_dict(cms)
    d = sort_collectors(d, 100)
    json_str = fix_float_for_json(d)

    f = open("collectors.json", "w+")
    f.write(json_str)
    f.close()


def load_compact_nouns_json():
    f = open("collectors.json", "r")
    x = f.read()
    f.close()
    return json.loads(x)

# _import_corpus()


x = load_compact_nouns_json()
# print (x["w1"]["litus"])
# print (x["w2"]["litus"])
# y = fix_float_for_json(x)

tokenizer = LineTokenizer('latin')
lemmatizer = BackoffLatinLemmatizer()

reader = get_corpus_reader(language='latin', corpus_name='latin_text_perseus')
docs = list(reader.docs())
# reader._fileids = ['cicero__on-behalf-of-aulus-caecina__latin.json']
sentences = list(reader.sents())

# to speedup
sentences = sentences[::1]

print (len(sentences))
# collection_method = DefaultCollectionMethod()
# collection_method = WordDistanceCollectionMethod(2, t_tsh=3, freq_tsh=0.01)
cm_1 = WordDistanceCollectionMethod(1, t_tsh=2, freq_tsh=0.01)
cm_2 = WordDistanceCollectionMethod(2, t_tsh=2, freq_tsh=0.01)
cm_4 = WordDistanceCollectionMethod(4, t_tsh=2, freq_tsh=0.01)
# cms = [cm_1, cm_2, cm_4]
cms = [cm_2]
cc = CollocationCollector(lemmatizer, None, cms)
cc.parse(sentences)
all_collocations = cc.find_collocation()
x = all_collocations[0]
# y = all_collocations[1]
# print (cc.g_counter["capio"])
print (x["litus"])
# print (y["litus"])
# print (y["musca"])
save_collectors(cms)
# very good examples in "stringo" "lacus" "iaceo corpus" "curnu"

# TODO: json with 2 numbers after the dot