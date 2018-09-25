# Contains code for reading from CSVs, normalizing text, and labeling text
import re
from random import shuffle
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from nltk.tokenize import sent_tokenize
from .util import extract_clinical_history, extract_findings, extract_impression

section_extraction_fns = {
    "impression": extract_impression,
    "clinical_history": extract_clinical_history,
    "findings": extract_findings,
}

drop_indicators = ["is a non-reportable study", "consent form", "informed consent"]
class SectionExtractor(TransformerMixin):
    def __init__(self, sections):
        self.extraction_fns = []
        for section in sections:
            if not section in section_extraction_fns.keys():
                raise KeyError("Unknown section " + section)
            self.extraction_fns.append(section_extraction_fns[section])

    def transform(self, reports, *_):
        result = []
        for report_obj in reports:
            if True in [di in report_obj["report"] for di in drop_indicators]:
                continue
            sections = [extract_fn(report_obj["report"]) for extract_fn in self.extraction_fns]
            report_obj["report"] = "\n".join(sections)
            if len(report_obj["report"]) > 0:
                result.append(report_obj)
        return result

punct = "!\"#$%&\'()*+,-.:;<=>?@[\]^_`{|}~\n"
class SentenceTokenizer(TransformerMixin):
    def transform(self, reports, *_):
        result = []
        for report_obj in reports:
            # Tokenize clinical history
            text = report_obj["ch"]
            text = text.replace("Dr.", "Dr")
            text = re.sub('[0-9]\. ', "", text)
            text = text.replace("r/o", "rule out")
            text = text.replace("R/O", "rule out")

            section_sentences = sent_tokenize(text)
            new_sentences = []
            for sentence in section_sentences:
                if len(sentence) <= 2:
                    continue
                for r in punct:
                    sentence = sentence.replace(r, " ")
                sentence = sentence.replace("/", " ")
                sentence = sentence.replace("  ", " ")
                sentence = sentence[:-1] if sentence[-1] == " " else sentence
                sentence = sentence.lower()
                new_sentences.append(sentence)
            report_obj["sentences"] = new_sentences
            result.append(report_obj)
        return result

class ReportObjCreator(TransformerMixin):
    def transform(self, report_tuples, *_):
        return [{"report" : rt, "exam" : ed} for rt, ed in report_tuples]

class ReportLabeler(TransformerMixin):
    def transform(self, reports, *_):
        result = []
        for report_obj in reports:
            label = fine_ground_truth_label(report_obj["exam"])
            if label != 2:
                result.append((report_obj["sentences"], label))
        return result

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
