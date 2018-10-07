import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from semantic_mapping import NegationMarker, DateTimeMapper, StopWordRemover

def test_negation():
    test_report = "there NEGEX evidence cancer. NEGEX fracture ribs, arms."
    test_in = np.reshape(np.array([test_report]), (-1, 1))
    out = NegationMarker().transform(test_in)
    assert(out == "there NEGEX_evidence NEGEX_cancer . NEGEX_fracture NEGEX_ribs , arms .")

def test_date_time_mapper():
    test_report = "event at 4:23pm on 4/23/18 for patient"
    test_in = np.reshape(np.array([test_report]), (-1, 1))
    out = DateTimeMapper.transform(test_in)
    print(out)
    assert(out == "event at TIME on DATE for patient")

def test_stop_word_removal():
    report = "Mild irregularity of the cavernous and supraclinoid internal carotid arteries related to atherosclerosis."
    test_in = np.reshape(np.array([report]), (-1, 1))
    out = StopWordRemover().transform(test_in)
    assert(out == "Mild irregularity cavernous supraclinoid internal carotid arteries related atherosclerosis .")