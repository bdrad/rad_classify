from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from nltk.tokenize import word_tokenize
from rad_classify import MapperTransformer, SentenceTransformer
import pickle
import re

def read_replacements(replacement_file_path):
    return pickle.load(open(replacement_file_path, 'rb'))

class SemanticMapper(SentenceTransformer):
    def __init__(self, replacements, regex=False, threads=1):
        self.replacements = replacements
        self.regex = regex
        self.threads = threads

    def sentence_map(self, sentence, *_):
        if self.regex:
            for r in self.replacements:
                sentence = re.sub(r[0], r[1], sentence)
            return sentence
        else:
            sentence = " " + sentence + " "
            for r in self.replacements:
                sentence = sentence.replace(r[0],r[1])
            sentence = sentence.replace("  ", " ").strip()
            return sentence

DateTimeMapper = SemanticMapper([(r'[0-9][0-9]?(/|-)[0-9][0-9]?(/|-)[0-9][0-9]([0-9][0-9])?', 'DATE'),
                                 (r'[0-9][0-9]?:[0-9][0-9] ?(am|pm)?', 'TIME')], regex=True)

AlphaNumRemover = SemanticMapper([(r' [0-9]+','')], regex=True)