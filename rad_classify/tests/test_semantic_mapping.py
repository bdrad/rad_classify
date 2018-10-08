import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from semantic_mapping import DateTimeMapper, SemanticMapper

def test_date_time_mapper():
    test_report = "event at 4:23pm on 4/23/18 for patient"
    test_in = np.reshape(np.array([test_report]), (-1, 1))
    out = DateTimeMapper.transform(test_in)
    print(out)
    assert(out == "event at TIME on DATE for patient")

def test_multithread_semantic_mapping():
    test_reports = [
        "abc",
        "cab", 
        "dab",
        "bac"
    ]
    test_in = np.reshape(np.array([test_reports]), (-1, 1))

    replacements = [('a', 'm'), ('d', 'e'), ('c', 'g'), ('b', 'f')]
    mapper = SemanticMapper(replacements, threads=4)

    out = mapper.transform(test_in)
    assert(list(out[:,0]) == ['mfg', 'gmf', 'emf', 'fmg'])

