# Contains code for reading from CSVs, normalizing text, and labeling text
import re
from random import shuffle
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from nltk.tokenize import sent_tokenize, word_tokenize
from .util import *

section_extraction_fns = {
    "impression": extract_impression,
    "clinical_history": extract_clinical_history,
    "findings": extract_findings,
}

drop_indicators = ["is a non-reportable study", "consent form", "informed consent"]
class SectionExtractor(MapperTransformer):
    def __init__(self, sections=None):
        self.extraction_fns = []
        if sections is None:
            self.extraction_fns = [lambda x: x] # Extract whole report
        else:
            for section in sections:
                if not section in section_extraction_fns.keys():
                    raise KeyError("Unknown section " + section)
                self.extraction_fns.append(section_extraction_fns[section])

    def map_fn(self, report, *_):
        print(report)
        if True in [di in report for di in drop_indicators]:
            return ""
        sections = [extract_fn(report) for extract_fn in self.extraction_fns]
        combined_report = "\n".join(sections)
        print(combined_report)
        return combined_report

class SentenceTokenizer(MapperTransformer):
    def map_fn(self, text, *_):
        text = text.replace("Dr.", "Dr")
        text = re.sub('[0-9]\. ', "", text)
        text = text.replace("r/o", "rule out")
        text = text.replace("R/O", "rule out")

        section_sentences = sent_tokenize(text)
        new_sentences = []
        for sentence in section_sentences:
            if len(sentence) <= 2:
                continue
            sentence = sentence.replace("/", " ")
            sentence = sentence.replace("  ", " ")
            sentence = sentence.strip().lower()
            new_sentences.append(sentence)
        return ". ".join(new_sentences)

punct = "!\"#$%&\'()*+,-.:;<=>?@[\]^`{|}~\n"
class PunctuationRemover(MapperTransformer):
    def map_fn(self, text, *_):
        words = word_tokenize(text)
        return ' '.join(c for c in words if c not in punct)


import argparse
import pickle
import itertools

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a corpus and output it to a file')
    parser.add_argument('-i','--in_path', nargs='+', required=True)
    parser.add_argument('-o','--out_path', nargs=1, required=True)

    args = parser.parse_args()

    data = [get_reports_from_csv(ip) for ip in args.in_path]
    merged_data = list(set(itertools.chain.from_iterable(data)))

    pipeline = make_pipeline(ReportObjCreator(), SectionExtractor(), SentenceTokenizer(), ReportLabeler(), None)
    preprocessed = pipeline.transform(merged_data)
    print("Writing " + str(len(preprocessed)) + " preprocessed reports")
    pickle.dump(preprocessed, open(args.out_path[0], "wb"))
