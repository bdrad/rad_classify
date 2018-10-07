from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Phrases
from rad_classify import MapperTransformer, SentenceTransformer
import pickle
import re

def read_replacements(replacement_file_path):
    return pickle.load(open(replacement_file_path, 'rb'))

negation_stopping_punct = ".,?!;:"
class NegationMarker(MapperTransformer):
    def map_fn(self, text, *_):
        words = word_tokenize(text)
        negating = False
        new_words = []
        for word in words:
            if word in negation_stopping_punct:
                negating = False
            if word == 'NEGEX':
                negating = True
            elif negating:
                new_words.append("NEGEX_" + word)
            else:
                new_words.append(word)
        return " ".join(new_words)

stop_words = set(stopwords.words('english'))
extra_removal = set(["cm", "mm", "x", "please", "is", "are", "be", "been"])
to_remove = stop_words.union(extra_removal)
class StopWordRemover(SentenceTransformer):
    def sentence_map(self, sentence, *_):
        words = word_tokenize(sentence)
        filtered_sentence = [w for w in words if (not w in to_remove) and (not w.isdigit())]
        return " ".join(filtered_sentence)

class SemanticMapper(SentenceTransformer):
    def __init__(self, replacements, regex=False):
        self.replacements = replacements
        self.regex = regex

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

import argparse
import pickle
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a corpus and output it to a file')
    parser.add_argument('labeled_reports_in_path')
    parser.add_argument('replacement_file_path')
    parser.add_argument('radlex_file_path')
    parser.add_argument('labels_out_path')
    args = parser.parse_args()

    labeled_reports = pickle.load(open(args.labeled_reports_in_path, "rb"))

    replacements = read_replacements(args.replacement_file_path)
    radlex_replacements = read_replacements(args.radlex_file_path)
    ReplacementMapper = SemanticMapper(replacements)
    RadlexMapper = SemanticMapper(radlex_replacements)
    pipeline = make_pipeline(DateTimeMapper, ExtenderPreserver, ReplacementMapper, RadlexMapper,
                             StopWordRemover(), ExtenderRemover, None)

    labeled_output = pipeline.transform(labeled_reports)

    pickle.dump(labeled_output, open(args.labels_out_path, "wb"))
