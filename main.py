from collections import defaultdict

from cltk.corpus.readers import get_corpus_reader
from cltk.lemmatize.latin.backoff import BackoffLatinLemmatizer
from cltk.tokenize.line import LineTokenizer

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
        self.counter = defaultdict(lambda: defaultdict(int))

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
        words = token.split()
        return self.lemmatizer.lemmatize(words)

    def parse_lemmas_in_group(self, grp):
        lemmas = {x[1] for x in grp}
        for l in lemmas:
            for l2 in lemmas-{l}:
                self.counter[l][l2] += 1


tokenizer = LineTokenizer('latin')
lemmatizer = BackoffLatinLemmatizer()

reader = get_corpus_reader(language='latin', corpus_name='latin_text_perseus')
docs = list(reader.docs())
# reader._fileids = ['cicero__on-behalf-of-aulus-caecina__latin.json']
sentences = list(reader.sents())
print (len(sentences))
cc = CollocationCollector(lemmatizer, tokenizer)
cc.parse(sentences)
print (cc.counter["capio"])
print ("test")


