import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from end2end_process import EndToEndPreprocessor

with open('./rad_classify/tests/data/example_report.txt') as report_file:
    example_report = report_file.read()
example_input = [example_report]

with open('./rad_classify/tests/data/fully_processed_report.txt') as report_file:
    expected_processed_report = report_file.read()

def test_end2end():
    processor = EndToEndPreprocessor(sections=['impression'])
    processed_report = processor.transform(example_input)[0]
    with open('./rad_classify/tests/data/fully_processed_report.txt') as report_file:
        expected_processed_report = report_file.read()
    assert(processed_report == expected_processed_report)