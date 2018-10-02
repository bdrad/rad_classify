# BDRAD Rad Classify
Rad Classify is a small Python library for quickly building classifiers for radiology reports. It leverages semantic dictionary mapping and fastText and is currently available as a black box classifier.

## Installation
Rad Classify depends on fastText for Python, which needs to be installed manually. Instructions for installation can be found on the fastText [GitHub page](https://github.com/facebookresearch/fastText#building-fasttext-for-python). The instructions are copied below for convenience:
```
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ pip install .
```
Once fastText is installed, you can install Rad Classify by cd'ing into the `rad_classify` directory and running
```
$ pip install .
```
Rad Classify also depends on NLTK and sklearn, but these should be installed by pip automatically.

## Usage
After installation, we can import the `rad_classify` module and start using its methods
```
from rad_classify import EndToEndProcessor, FastTextClassifier, get_reports_from_csv

path = "./data/rad_reports.csv"
reports, labels = zip(*get_reports_from_csv(path, report_col="Report Text", label_col="Label"))
```
`get_reports_from_csv` is a utility method for reading CSVs. The `EndToEndProcessor` performs all of the necessary preprocessing before the text is fed into the classify. You can pass in paths to semantic dictionary files for performing semantic dictionary mapping. These files should be pickled dictionaries of strings to their replacements. [CLEVER](https://github.com/stamang/CLEVER) and [RadLex](http://radlex.org/) dictionaries are provided with the library.
```
preprocessor = EndToEndProcessor(replacement_file_path="./semantic_dictionaries/clever_replacements", radlex_path="./semantic_dictionaries/radlex_replacements", sections=None, sections=["impression"])
processed_reports = preprocessor.transform(reports)
```
Note that you provide which section(s) of the radiology report you wish to extract and use for classication. Options are `impression`, `findings`, and `clinical_history`. Once we've processed the reports, we can use fastText to classify them.
```
clf = rad_classify.FastTextClassifier()
clf.train(processed_reports, labels, dim=50, epoch=20, lr=0.05)
```
Now that we've trained the classifier, we can use it to classify other reports
```
valid_path = "./data/validation_set.csv"
validation_reports, validation_labels = zip(*get_reports_from_csv(path, report_col="Report Text", label_col="Label"))
processed_valid = preprocessor.transform(validation_reports)
predictions = clf.predict(processed_valid)
```
