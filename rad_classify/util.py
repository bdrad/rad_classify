import re
import csv
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.base import TransformerMixin

class MapperTransformer(TransformerMixin):
    '''
    sklearn transformer which applies map_fn to each sample in X
    '''
    def map_fn(self, x):
        return report

    def transform(self, X, *_):
        return np.vectorize(self.map_fn)(X)

class SentenceTransformer(MapperTransformer):
    '''
    Transformer which applies sentence_map_fn to each sentence in a string
    '''
    def sentence_map(self, sentence):
        return sentence

    def map_fn(self, report_text):
        sentences = sent_tokenize(report_text)
        mapped_sentences = [self.sentence_map(s) for s in sentences]
        return " ".join(mapped_sentences)


def extract_impression(report_text):
    '''
    report_text: (str) A standard ucsf report text associated with an record entry.
    Returns: (str) The Impression section from the report
    '''
    im_search = re.search('IMPRESSION:((.|\n)+?)(END OF IMPRESSION|Report dictated by:|\/\/|$)', report_text)
    if im_search:
        return im_search.group(1).strip().replace(' \n', '\n')
    else:
        return ""

def extract_clinical_history(report_text):
    '''
    report_text: (str) A standard ucsf report text associated with an record entry.
    Returns: (str) The clinical history from the report
    '''
    ch_search = re.search('CLINICAL HISTORY:((.|\n)+?)\n([A-Z]| )+:', report_text)
    if ch_search:
        return ch_search.group(1).strip().replace(' \n', '\n')
    else:
        return ""

def extract_findings(report_text):
    '''
    report_text: (str) A standard ucsf report text associated with an record entry.
    Returns: (str) The Findings section from the report
    '''
    ch_search = re.search('FINDINGS:((.|\n)+?)\n([A-Z]| )+:', report_text)
    if ch_search:
        return ch_search.group(1).strip().replace(' \n', '\n')
    else:
        return ""

def get_reports_from_csv(csv_path, report_col='Report Text', label_col='Label'):
    '''
    csv_path: (str) Path to CSV which contains radiology reports
    report_col: (str) Name of column which contains the report text
    label_col: (str) Name of column which contains the label for classification
    Returns: Iterable[(str, str)] iterable of tuples of (Report Text, Label)
    '''
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        while True:
            n = next(reader, None)
            if n is None:
                return
            yield (n[report_col], n[label_col])
