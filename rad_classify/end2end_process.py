from preprocessing import SectionExtractor, SentenceTokenizer, ReportLabeler, ReportObjCreator
from semantic_mapping import DateTimeMapper, SemanticMapper, StopWordRemover, NegexSmearer, ExtenderPreserver, ExtenderRemover
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer
import pickle

def read_replacements(replacement_file_path):
    return pickle.load(open(replacement_file_path, 'rb'))

class EndToEndProcessor(TransformerMixin):
    def __init__(self, replacement_file_path=None, radlex_path=None, sections=["impression"]):
        if replacement_file_path is  None:
            ReplacementMapper = FunctionTransformer()
        else:
            replacements = read_replacements(replacement_file_path)
            ReplacementMapper = SemanticMapper(replacements)

        if radlex_path is None:
            RadlexMapper = FunctionTransformer()
        else:
            radlex_replacements = read_replacements(radlex)
            RadlexMapper = SemanticMapper(radlex_replacements)

        self.pipeline = make_pipeline(ReportObjCreator(),SectionExtractor(sections),
            SentenceTokenizer(), ReportLabeler(), DateTimeMapper,
            ExtenderPreserver, ReplacementMapper, RadlexMapper,
            StopWordRemover(), NegexSmearer(), ExtenderRemover, None)

    def transform(self, reports, *_):
        return self.pipeline.transform(reports)
