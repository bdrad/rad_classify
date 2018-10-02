from .preprocessing import SectionExtractor, SentenceTokenizer, PunctuationRemover
from .semantic_mapping import *
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer
import pickle
import numpy as np

def read_replacements(replacement_file_path):
    return pickle.load(open(replacement_file_path, 'rb'))

class EndToEndProcessor():
    def __init__(self, replacement_path="./rad_classify/semantic_dictionaries/clever_replacements", radlex_path="./rad_classify/semantic_dictionaries/radlex_replacements", sections=None):
        if replacement_path is  None:
            ReplacementMapper = FunctionTransformer()
        else:
            replacements = read_replacements(replacement_path)
            ReplacementMapper = SemanticMapper(replacements)

        if radlex_path is None:
            RadlexMapper = FunctionTransformer()
        else:
            radlex_replacements = read_replacements(radlex_path)
            RadlexMapper = SemanticMapper(radlex_replacements)

        self.pipeline = make_pipeline(SectionExtractor(sections),
            SentenceTokenizer(), DateTimeMapper, ReplacementMapper,
            RadlexMapper, StopWordRemover(), NegationMarker(),
            PunctuationRemover(), None)

    def transform(self, reports):
        report_array = np.reshape(np.array(reports), (-1, 1))
        transformed_array = self.pipeline.transform(report_array)
        return transformed_array[:,0]
