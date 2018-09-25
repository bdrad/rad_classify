from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Phrases
import pickle
import re

def read_replacements(replacement_file_path):
    return pickle.load(open(replacement_file_path, 'rb'))

def reports_to_corpus(reports, out_file):
    for report in reports:
        for sentence in report[0]:
            out_file.write(sentence + "\n")

class NegexSmearer(TransformerMixin):
    def __init__(self, negex_range=5):
        self.negex_range = negex_range
    def transform(self, labeled_reports, *_):
        return labeled_reports # REMOVE FOR NEGEX
        result = []
        if self.negex_range == 1:
            for labeled_report in labeled_reports:
                new_report = []
                for sentence in labeled_report[0]:
                    new_report.append(sentence.replace("NEGEX ", "NEGEX_").replace("NEGEX_NEGEX_", "NEGEX_"))
                result.append((new_report, labeled_report[1]))
        else:
            for labeled_report in labeled_reports:
                report = labeled_report[0]
                new_sentences = []
                for sentence in report:
                    tokenized = sentence.split(" ")
                    negex_indices = [i for i, t in enumerate(tokenized) if t == "NEGEX"]
                    to_negate = []
                    for offset in range(1, self.negex_range + 1):
                        to_negate += [i + offset for i in negex_indices]
                    to_negate = [tn for tn in to_negate if not tn in negex_indices]
                    to_negate = [tn for tn in to_negate if tn < len(tokenized) and tn >= 0]

                    EXT_indices = []
                    for i in to_negate:
                        if tokenized[i] == "EXT":
                            EXT_indices.append(i)
                            to_negate = to_negate + [r + i for r in range(1, self.negex_range + 1)]
                    to_negate = [tn for tn in to_negate if not (tn in negex_indices or tn in EXT_indices)]
                    to_negate = [tn for tn in to_negate if tn < len(tokenized) and tn >= 0]

                    for i in to_negate:
                        tokenized[i] = "NEGEX_" + tokenized[i]
                    clean_tokenized = [t for i, t in enumerate(tokenized) if not i in negex_indices]
                    new_sentences.append(" ".join(clean_tokenized).replace("NEGEX_NEGEX_", "NEGEX_"))
                result.append((new_sentences, labeled_report[1]))

        return result

class PhraseDetector(TransformerMixin):
    def transform(self, labeled_reports, *_):
        # Build bigram detector
        sentences = []
        for labeled_report in labeled_reports:
            report = labeled_report[0]
            sentences += [sent.split(" ") for sent in report]
        bigram = Phrases(sentences, scoring="npmi", threshold=0.45)

        # Replace bigrams
        result = []
        for labeled_report in labeled_reports:
            report = labeled_report[0]
            new_report = [" ".join(bigram[sentence.split(" ")]) for sentence in report]
            result.append((new_report, labeled_report[1]))
        return result

stop_words = set(stopwords.words('english'))
extra_removal = set(["cm", "mm", "x", "please", "is", "are", "be", "been"])
to_remove = stop_words.union(extra_removal)
class StopWordRemover(TransformerMixin):
    def transform(self, labeled_reports, *_):
        result = []
        for report in labeled_reports:
            new_sentences = []
            for sentence in report[0]:
                words = word_tokenize(sentence)
                filtered_sentence = [w for w in words if (not w in to_remove) and (not w.isdigit())]
                new_sentences.append(" ".join(filtered_sentence))
            result.append((new_sentences, report[1]))
        return result


class SemanticMapper(TransformerMixin):
    def __init__(self, replacements, regex=False):
        self.replacements = replacements
        self.regex = regex

    def transform_regex(self, labeled_report):
        result = []
        for report in labeled_report:
            new_sentences = []
            for sentence in report[0]:
                for r in self.replacements:
                    sentence = re.sub(r[0], r[1], sentence)
                new_sentences.append(sentence)
            result.append((new_sentences, report[1]))
        return result

    def transform(self, labeled_report, *_):
        if self.regex:
            return self.transform_regex(labeled_report)

        result = []
        for report in labeled_report:
            new_sentences = []
            for sentence in report[0]:
                sentence = " " + sentence + " "
                for r in self.replacements:
                    sentence = sentence.replace(r[0],r[1])
                sentence = sentence.replace("  ", " ")
                if len(sentence) > 1 and sentence[0] == " ":
                    sentence = sentence[1:]
                if len(sentence) > 1 and sentence[-1] == " ":
                    sentence = sentence[:-1]
                new_sentences.append(sentence)
            result.append((new_sentences, report[1]))
        return result

DateTimeMapper = SemanticMapper([(r'[0-9][0-9]? [0-9][0-9]? [0-9][0-9][0-9][0-9]', ''),
                                 (r'[0-9][0-9]? [0-9][0-9] (am|pm)?', '')], regex=True)

AlphaNumRemover = SemanticMapper([(r' [0-9]+','')], regex=True)

ExtenderPreserver = SemanticMapper([(' or ', ' EXT '), (' nor ', ' EXT ')])
ExtenderRemover = SemanticMapper([('EXT', ''), ('NEGEX_EXT', ''), (('NEGEX_ ', ''))])

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
