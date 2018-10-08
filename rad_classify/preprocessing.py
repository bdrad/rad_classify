# Contains code for reading from CSVs, normalizing text, and labeling text
import re
from random import shuffle
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from rad_classify import extract_impression, extract_clinical_history, extract_findings, MapperTransformer, SentenceTransformer

section_extraction_fns = {
    "impression": extract_impression,
    "clinical_history": extract_clinical_history,
    "findings": extract_findings,
}

drop_indicators = ["is a non-reportable study", "consent form", "informed consent"]
class ClearDroppedReports(MapperTransformer):
    def map_fn(self, report):
        if any(di in report.lower() for di in drop_indicators):
            return ""
        return report

class SectionExtractor(MapperTransformer):
    def __init__(self, sections=None, threads=1):
        self.threads = threads
        self.extraction_fns = []
        if sections is None:
            self.extraction_fns = [lambda x: x] # Extract whole report
        else:
            for section in sections:
                if not section in section_extraction_fns.keys():
                    raise KeyError("Unknown section " + section)
                self.extraction_fns.append(section_extraction_fns[section])

    def map_fn(self, report, *_):
        sections = [extract_fn(report) for extract_fn in self.extraction_fns]
        return "\n".join(sections)

class SentenceTokenizer(MapperTransformer):
    def map_fn(self, text, *_):
        text = text.replace("Dr.", "Dr")
        text = text.replace("r/o", "rule out")
        text = text.replace("R/O", "rule out")

        section_sentences = sent_tokenize(text)
        new_sentences = []
        for sentence in section_sentences:
            if len(sentence) <= 2:
                continue
            sentence = sentence.replace("/", " ")
            sentence = sentence.replace(":", " ")
            sentence = sentence.replace("  ", " ")
            sentence = sentence.strip().lower()
            new_sentences.append(sentence)
        return " ".join(new_sentences)

punct = '!"#$%&\'()*+,-.:;<=>?@[\]^`{|}~\n'
class PunctuationRemover(MapperTransformer):
    def map_fn(self, text, *_):
        words = word_tokenize(text)
        return ' '.join(c for c in words if c not in punct)


negation_stopping_punct = ".,?!;:"
class NegationMarker(MapperTransformer):
    def __init__(self, negation_phrases=["NEGEX"], threads=1):
        self.negation_phrases = negation_phrases
        self.threads = threads

    def map_fn(self, text, *_):
        words = word_tokenize(text)
        negating = False
        new_words = []
        for word in words:
            if word in negation_stopping_punct:
                negating = False
            if word in self.negation_phrases:
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