import json
from json import JSONEncoder

import math
from collections import defaultdict

from cltk.corpus.readers import get_corpus_reader
from cltk.lemmatize.latin.backoff import BackoffLatinLemmatizer
from cltk.stem.latin.j_v import JVReplacer
from cltk.tokenize.line import LineTokenizer

import re
import gc

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
        self.name = "no name"

    def find(self):
        self.pairs = self._get_pairs()
        # print("Here:{}".format(self.get_name()))
        self.ref = self._get_ref()
        # print("Here2:{}".format(self.get_name()))
        return self.pairs

    def _get_pairs(self):
        return {}

    def _get_ref(self):
        return {}

    def get_name(self):
        return self.name

    def save(self, path=""):
        d = {'pairs': self.pairs,
             'ref': self.ref,
             'name': self.name}
        json_str = json.dumps(d)
        path += "{}.json".format(self.name)
        f = open(path, "w+")
        f.write(json_str)
        f.close()

    @staticmethod
    def load(path):
        res = AbstractCollectionMethod()
        f = open(path, "r")
        x = f.read()
        f.close()

        d = json.loads(x)
        res.name = d['name']
        res.pairs = d['pairs']
        res.ref = d['ref']

        return res

    @staticmethod
    def extract_pairs_from_data(pairs_data, number_of_groups, lemmas_count, min_occurrences,
                                t_tsh, max_freq=7777.0):
        lemmas_freq = {l: float(lemma_count) / number_of_groups for l, lemma_count in lemmas_count.items()}
        pairs = defaultdict(lambda: defaultdict(int))
        for l, l_group in pairs_data.items():
            for l2, l2_count in l_group.items():
                if lemmas_count[l] >= min_occurrences and lemmas_count[l2] >= min_occurrences:
                    if lemmas_freq[l] < max_freq and lemmas_freq[l2] < max_freq:
                        t = AbstractCollectionMethod.analyze_pair(l, l2, pairs_data, lemmas_freq, number_of_groups)
                        if t >= t_tsh:
                            pairs[l][l2] = t
        return pairs

    @staticmethod
    def analyze_pair(l1, l2, pairs_data, lemmas_freq_in_group, groups_count):
        p1 = lemmas_freq_in_group[l1]
        p2 = lemmas_freq_in_group[l2]
        m = p1 * p2
        x = float(pairs_data[l1][l2]) / groups_count
        s2 = x * (1 - x)
        # print (p1,p2,m,x,s2,groups_count)
        t = (x - m) / math.sqrt(s2 / groups_count)
        return t



class DefaultCollectionMethod(AbstractCollectionMethod):
    def __init__(self, t_tsh=4.0, freq_tsh=0.1, min_occurrences=10):
        super(AbstractCollectionMethod).__init__()
        self.g_counter = defaultdict(lambda: defaultdict(int))
        self.lemma_counter = defaultdict(int)
        self.total_groups = 0
        self.t_tsh = t_tsh
        self.freq_tsh = freq_tsh
        self.min_occurrences = min_occurrences
        self._ref = defaultdict(lambda: defaultdict(int))
        self.grp_list = []
        # self.all_pairs = set()

    def parse_lemmas_in_group(self, grp):
        grp_dict = {x[1]: x[0] for x in grp}
        # due to space issue we should keep the entire grp,
        # and later to run over the required paris only
        self.grp_list.append(grp_dict)

        lemmas = grp_dict.keys()
        for l,v in grp_dict.items():
            self.lemma_counter[l] += 1
            self._ref[l][v] += 1
            for l2 in lemmas-{l}:
                self.g_counter[l][l2] += 1
                # frozenset to avoid duplicats. in comment (see above why)
                #self._ref[frozenset((l,l2))].add(frozenset((grp_dict[l],grp_dict[l2])))
        self.total_groups += 1

    def _get_ref(self):

        return self._ref

        # all_head_words = self.g_counter.keys()
        # for grp in self.grp_list:
        #     for k, v in grp.items():
        #         if k in all_head_words:
        #             self._ref[k][v] += 1


        # In comment due to space/time issues
        # for pair in self.all_pairs:
        #     [l1, l2] = list(pair)
        #     for grp in self.grp_list:
        #         lemmas_set = set(grp.keys())
        #         if pair <= lemmas_set:
        #             self._ref[pair].add(frozenset((grp[l1], grp[l2])))
        # return self._ref

    def _get_pairs(self):


        return AbstractCollectionMethod.extract_pairs_from_data(
            pairs_data=self.g_counter,
            number_of_groups=self.total_groups,
            lemmas_count=self.lemma_counter,
            min_occurrences=self.min_occurrences,
            t_tsh=self.t_tsh,
            max_freq=self.freq_tsh
        )

        # pairs = defaultdict(lambda: defaultdict(int))
        # lemmas_freq_in_group = {l: float(self.lemma_counter[l]) / self.total_groups for l in self.lemma_counter}
        # lemmas_occurrences = {l: sum(occ.values()) for l,occ in self._ref.items()}
        # for l, l_group in self.g_counter.items():
        #     for paired_l, paried_l_count in l_group.items():
        #         if (lemmas_freq_in_group[l] < self.freq_tsh and lemmas_freq_in_group[paired_l] < self.freq_tsh) \
        #                 and (lemmas_occurrences[l] >= self.min_occurrences and lemmas_occurrences[paired_l] >= self.min_occurrences):
        #             t = self._analyze_pair(l, paired_l, lemmas_freq_in_group)
        #             if t > self.t_tsh:
        #                 # print(l, paired_l, t)
        #                 pairs[l][paired_l] = t
        #                 # self.all_pairs.add(frozenset((l, paired_l)))
        # return pairs

    # def _analyze_pair(self, l1, l2, lemmas_freq_in_group):
    #     p1 = lemmas_freq_in_group[l1]
    #     p2 = lemmas_freq_in_group[l2]
    #     m = p1 * p2
    #     x = float(self.g_counter[l1][l2]) / self.total_groups
    #     s2 = x * (1 - x)
    #     t = (x - m) / math.sqrt(s2 / self.total_groups)
    #     return t

class RandomSliceCollectionMethod(AbstractCollectionMethod):
    def __init__(self, slice_size, t_tsh=4.0, freq_tsh=0.1, min_occurrences=10):
        super(AbstractCollectionMethod).__init__()
        self.slice_size = slice_size
        self.name = "{}{}".format("r", str(self.slice_size))
        self.how_many_slices_was_the_middle_count_in = []
        self.min_occurrences = min_occurrences
        self._ref = defaultdict(lambda: defaultdict(int))
        self.default_collection = DefaultCollectionMethod(t_tsh=t_tsh, freq_tsh=freq_tsh,
                                                          min_occurrences=0)

    def parse_lemmas_in_group(self, grp):
        # We want groups in slices (not included):
        # (0, slice_size)
        # (1, slice_size+1)
        _current_number_of_slices = 0
        middle_inedx = len(grp) // 2
        grp_fixed_list = list(grp)
        for i, l in enumerate(grp):
            self._ref[l[1]][l[0]] += 1
            start = i
            end = min(len(grp), i + self.slice_size)
            _slice = grp_fixed_list[start:end]
            self.default_collection.parse_lemmas_in_group(_slice)
            if middle_inedx >= start and middle_inedx < end:
                _current_number_of_slices += 1
            if end == len(grp):
                break
        self.how_many_slices_was_the_middle_count_in.append(_current_number_of_slices)

    def _get_pairs(self):
        _avg = sum(self.how_many_slices_was_the_middle_count_in) / float(len(self.how_many_slices_was_the_middle_count_in))
        self.default_collection.min_occurrences = self.min_occurrences * _avg
        return self.default_collection._get_pairs()

    def _get_ref(self):
        return self._ref


class WordDistanceCollectionMethod(AbstractCollectionMethod):
    def __init__(self, word_distance, t_tsh=4.0, freq_tsh=0.1, min_occurrences=10):
        super(AbstractCollectionMethod).__init__()
        self.word_distance = word_distance
        self.name = "{}{}".format("w", str(self.word_distance))
        self.min_occurrences = min_occurrences
        self.t_tsh = t_tsh
        self.pairs_counter = defaultdict(lambda: defaultdict(int))
        self.lemma_counter = defaultdict(int)
        self.slice_sizes_per_lemma = defaultdict(list)
        self._ref = defaultdict(lambda: defaultdict(int))

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
            self._ref[l[1]][l[0]] += 1
            self.lemma_counter[l[1]] += 1
            start = max(0, i - self.word_distance)
            end = min(len(grp) - 1, i + self.word_distance)
            _slice = grp_fixed_list[start:end + 1]
            self.slice_sizes_per_lemma[l[1]].append(len(_slice))
            lemma_index_in_slice = i - start
            self._parse_slice(_slice, lemma_index_in_slice)

    def _parse_slice(self, slice, lemma_index):
        lemma = slice[lemma_index][1]
        del slice[lemma_index]
        # There is a question if we should count a lemma more than once,
        # if it occurs so : YXY - I think it should be counted indeed
        # because eventually the computation is the number of co-occurneces
        # devided by chance it happens statisically.
        # on the other hand: meaning-base approuch says that a word cannot
        # be connected twice.
        # finally I think it should be counted once! since what I'm counting
        # really is pairs: different pairs. XYX - contains one pair.
        # hence we are using set
        lemmas_in_slice = {x[1] for x in slice}
        for cl in lemmas_in_slice:
            self.pairs_counter[lemma][cl] += 1

    def _get_pairs(self):
        # question: should account for the size size that is not always 2*word_distance?
        # but many times much less.
        # it makes no sense to me that the t-value for (x,y) and (y,x) will be different.
        # I think pure random will simply take the mean slice size for all lemmas
        number_of_slices = sum([len(x) for x in self.slice_sizes_per_lemma.values()])
        total_slice_size = sum([sum(x) for x in self.slice_sizes_per_lemma.values()])
        # slice slice includes our main lemma, hence 1 should be sub
        mean_slice_size = float(total_slice_size) / number_of_slices
        mean_slice_size -= 1
        total_lemmas = sum(self.lemma_counter.values())
        number_of_groups = float(total_lemmas) / mean_slice_size
        # lemma_freq = {l: float(lemma_count) * mean_slice_size / total_lemmas for l, lemma_count in self.lemma_counter.items()}
        # lemma_freq = lemma_count / number_of_groups
        # lemma_freq = {l: float(lemma_count) / number_of_groups for l, lemma_count in self.lemma_counter.items()}
        return AbstractCollectionMethod.extract_pairs_from_data(
            pairs_data=self.pairs_counter,
            number_of_groups=number_of_groups,
            lemmas_count=self.lemma_counter,
            min_occurrences=self.min_occurrences,
            t_tsh=self.t_tsh
        )

    def _get_ref(self):
        return self._ref



class CollocationCollector():
    def __init__(self, lemmatizer, tokenizer, collection_methods):
        self.lemmatizer = lemmatizer
        self.tokenizer = tokenizer
        self.collection_methods = collection_methods
        self.jv_replacer = JVReplacer()
        # self.g_counter = defaultdict(lambda: defaultdict(int))
        # self.lemma_counter = defaultdict(int)
        # self.total_groups = 0

    def parse(self, sentences):
        for i, s in enumerate(sentences):
            if 0 == (i % 10000):
                print ("sentences: {}".format(i))
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
        bad_lemmas = {"", "p"}
        # i,j and lower
        token = token.lower()
        token = self.jv_replacer.replace(token)
        regex = re.compile('[^a-zA-Z]')
        words = [regex.sub('', x) for x in token.split()]
        lemmas = self.lemmatizer.lemmatize(words)
        # BUG: self.lemmatizer.lemmatize(['uultuque'])-> vultue
        # i.e if ends with "que", it return "v" instead of "u"
        # so we need to run replace again
        return [(x[0], self.jv_replacer.replace(x[1])) for x in lemmas if x[1] not in bad_lemmas]

    def find_sentences(self, required_lemmas, sentences):
        for s in sentences:
            if self.is_sentence_have_all_lemmas(s, required_lemmas):
                print (s)

    def is_sentence_have_all_lemmas(self, s, required_lemmas):
        d = self.sentence_to_lemmas_dict(s)
        for rl, rv in required_lemmas.items():
            if d[rl] < rv:
                return False
        return True

    def sentence_to_lemmas_dict(self, sen):
        l_t = self.lemmatize_token(sen)
        res = defaultdict(int)
        for x in l_t:
            res[x[1]] += 1

        return res

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

def arrange_by_lemma(d):
    all_lemmas = set()
    new_dict = {}

    for cm, dv in d.items():
        all_lemmas = all_lemmas.union(set(dv.keys()))

    for l in all_lemmas:
        new_dict[l] = {}
        for cm, dv in d.items():
            d_value_for_l = dv.get(l, {})
            new_dict[l][cm] = d_value_for_l

    return new_dict

def save_collectors(cms, should_arrange_by_lemma=True):
    d = merge_collection_methods_to_dict(cms)
    d = sort_collectors(d, 100)
    if should_arrange_by_lemma:
        d = arrange_by_lemma(d)
    json_str = fix_float_for_json(d)

    f = open("collectors.json", "w+")
    f.write(json_str)
    f.close()

    save_ref(cms, set(d.keys()))

def merge_refs(refs):
    merged = defaultdict(lambda: defaultdict(int))
    for ref in refs:
        for k, dv in ref.items():
            for v, value in dv.items():
                merged[k][v] += value

    return merged

def calculate_ref(ref):
    calculated = {}
    for k, v in ref.items():
        l = [(_k, _v) for _k, _v in sorted(v.items(), key=lambda item: item[1], reverse=True)]
        # calculated[k] = l[:2]
        sugg = []
        find_greedy_suggestions(l, sugg, required_suggestion_count=2)
        calculated[k] = sugg

    # for k, v in ref.items():
    #     find_greedy_suggestions(v)

    return calculated

def find_greedy_suggestions(l, suggestions, required_suggestion_count=3):
    sug1 = ""
    usage_percentage = 1
    first_word = l[0][0]
    # Ideally, I would like "limit" to be function of the number of suggestion left.
    # the higher the number the higher the limit.
    # but this optimization is on low-priority
    limit = 0.33
    while usage_percentage > limit:
        if len(sug1) == len(first_word):
            break
        sug1 = first_word[0: len(sug1) + 1]
        usage_percentage = get_usage_percentage(sug1, l)

    if (usage_percentage <= limit) and len(sug1) > 1:
        sug1 = sug1[0:-1]
    if sug1 not in suggestions:
        suggestions.append(sug1)

        if len(suggestions) == required_suggestion_count:
            return
    else:
        return

    f_l = [x for x in l if not x[0].startswith(sug1)]
    if not f_l:
        return
    return find_greedy_suggestions(f_l, suggestions)

def get_usage_percentage(start, l):
    total = sum([_x[1] for _x in l])

    filtered = [_x for _x in l if _x[0].startswith(start)]
    usage = sum([_x[1] for _x in filtered])

    return float(usage) / total

def save_ref(cms, lemmas):
    merged = merge_refs([x.ref for x in cms])
    ref = calculate_ref(merged)

    ref = {x:v for x,v in ref.items() if x in lemmas}

    json_str = json.dumps(ref)
    f = open("ref.json", "w+")
    f.write(json_str)
    f.close()


def load_collectors_json():
    f = open("collectors.json", "r")
    x = f.read()
    f.close()
    return json.loads(x)

def load_ref_json():
    f = open("ref.json", "r")
    x = f.read()
    f.close()
    return json.loads(x)

def create_cm_list(wd, rs):
    l = []
    for w in wd:
        l.append(WordDistanceCollectionMethod(w, t_tsh=1.8))
    for r in rs:
        l.append(RandomSliceCollectionMethod(r, t_tsh=2, freq_tsh=0.01))

    return l

def load_cms_from_path(l):
    return [AbstractCollectionMethod.load(x) for x in l]

def run_cms_one_by_one(_cms, _lemmatizer):
    while len(_cms):
        cm = _cms.pop()
        _cc = CollocationCollector(_lemmatizer, None, [cm])
        _cc.parse(sentences)
        _cc.find_collocation()
        cm.save()
        cm = None
        gc.collect()


# _import_corpus()


x = load_collectors_json()
z = load_ref_json()

try:
    print(x["litus"])
except Exception as e:
    print ("no litus")
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
# cm_4 = WordDistanceCollectionMethod(4, t_tsh=2, freq_tsh=0.01)
# cm_8 = WordDistanceCollectionMethod(8, t_tsh=2, freq_tsh=0.01)
# rw_4 = RandomSliceCollectionMethod(4, t_tsh=2, freq_tsh=0.01)
# rw_8 = RandomSliceCollectionMethod(8, t_tsh=2, freq_tsh=0.01)
# rw_16 = RandomSliceCollectionMethod(16, t_tsh=2, freq_tsh=0.01)
# rw_2 = RandomSliceCollectionMethod(2, t_tsh=2, freq_tsh=0.01)
# cms = [cm_1, cm_2]
cms = create_cm_list(wd=[1, 2, 3, 4, 6, 8], rs=[1, 2, 3, 4, 6, 8, 12, 16])
cc = CollocationCollector(lemmatizer, None, cms)
# sentences = ["nec pedes nec caput"]
# cc.find_sentences({"haereo": 1, "lutum": 1}, sentences)
# raise Exception("Fdfd")
print ("start")
# cc.parse(sentences)
# all_collocations = cc.find_collocation()
# x = all_collocations[0]
# print (x["litus"])


# We run one by one due to memory issues
# run_cms_one_by_one(cms, lemmatizer)
files = [x.get_name() + ".json" for x in cms]
cms = load_cms_from_path(files)
save_collectors(cms)
# very good examples in "stringo" "lacus" "iaceo corpus" "curnu"

# TODO: json with 2 numbers after the dot