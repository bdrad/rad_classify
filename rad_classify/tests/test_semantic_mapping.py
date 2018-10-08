import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from semantic_mapping import DateTimeMapper

def test_date_time_mapper():
    test_report = "event at 4:23pm on 4/23/18 for patient"
    test_in = np.reshape(np.array([test_report]), (-1, 1))
    out = DateTimeMapper.transform(test_in)
    print(out)
    assert(out == "event at TIME on DATE for patient")
