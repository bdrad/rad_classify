import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import rad_classify
#from util import extract_impression, extract_clinical_history, extract_findings

with open('./rad_classify/tests/data/example_report.txt') as report_file:
    example_report = report_file.read()

def test_impression_extraction():
    impression = rad_classify.util.extract_impression(example_report)
    expected_impression = open('./rad_classify/tests/data/impression.txt').read()
    assert(impression == expected_impression)

def test_ch_extraction():
    clinical_history = rad_classify.util.extract_clinical_history(example_report)
    extracted_clinical_history = open('./rad_classify/tests/data/clinical_history.txt').read()
    assert(clinical_history == extracted_clinical_history)

def test_extract_findings():
    findings = rad_classify.util.extract_findings(example_report)
    expected_findings = open('./rad_classify/tests/data/findings.txt').read()
    assert(findings == expected_findings)