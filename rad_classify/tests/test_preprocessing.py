import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing import SectionExtractor, SentenceTokenizer, ClearDroppedReports, PunctuationRemover, StopWordRemover, NegationMarker

with open('./rad_classify/tests/data/example_report.txt') as report_file:
    example_report = report_file.read()
example_input = np.reshape(np.array([example_report]), (-1, 1))

def test_identity_section_extractor():
    extractor = SectionExtractor()
    output = extractor.transform(example_input)
    assert(output == example_input)

def test_impression_section_extractor():
    extractor = SectionExtractor(sections=['impression'])
    output = extractor.transform(example_input)
    expected_impression = open('./rad_classify/tests/data/impression.txt').read()
    assert(output[0] == expected_impression)

def test_dropping():
    dropped_report = "This is a non-reportable study.\n" + example_report
    dropped_input = np.reshape(np.array([dropped_report]), (-1, 1))
    dropper = ClearDroppedReports()
    output = dropper.transform(dropped_input)
    assert(output[0] == "")

def test_sentence_tokenization():
    test_in = "Report dicussed with Dr. X, r/o cancer. Measurement 9/3 mm."
    tokenizer = SentenceTokenizer()
    output = tokenizer.transform(test_in)
    assert(output == "report dicussed with dr x, rule out cancer. measurement 9 3 mm.")

def test_negation():
    test_report = "there NEGEX evidence cancer. NEGEX fracture ribs, arms."
    test_in = np.reshape(np.array([test_report]), (-1, 1))
    out = NegationMarker().transform(test_in)
    assert(out == "there NEGEX_evidence NEGEX_cancer . NEGEX_fracture NEGEX_ribs , arms .")

def test_stop_word_removal():
    report = "Mild irregularity of the cavernous and supraclinoid internal carotid arteries related to atherosclerosis."
    test_in = np.reshape(np.array([report]), (-1, 1))
    out = StopWordRemover().transform(test_in)
    assert(out == "Mild irregularity cavernous supraclinoid internal carotid arteries related atherosclerosis .")