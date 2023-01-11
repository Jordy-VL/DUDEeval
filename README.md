# DUDEeval 

This repository will help participants of DUDE @ICDAR2023 to set up their submission and validate results on sample data. 

## Installation

The scripts require [python >= 3.8](https://www.python.org/downloads/release/python-380/) to run.
We will create a fresh virtualenvironment in which to install all required packages.
```sh
mkvirtualenv -p /usr/bin/python3 DUDEeval
```

Using poetry and the readily defined pyproject.toml, we will install all required packages
```sh
workon DUDEeval 
pip3 install poetry
poetry install
```

## Way of working

Open a terminal in the directory and run the command:
```python
python3 evaluate_submission.py -g=gt/DUDE_demo_gt.json -s=submissions/DUDE_demo_submission_perfect.json
```

### parameters:

-g: Path of the Ground Truth file. The Ground Truth file is the one provided for the competition. You will be able to get it on the Downloads page of the Task in the Competition portal.

-s: Path of your method's results file.
 
#### Optional parameters:

-t: ANLS threshold. By default 0.5 is used. This can be used to check the tolerance to OCR errors. See Scene-Text VQA paper for more info.

-a: Boolean to get the scores break down by types. The ground truth file is required to have such information (currently is not available to the public).

-o: Path to a directory where to copy the file 'results.json' that contains per-sample results.

-c: Measure ECE for scoring calibration of answer confidences

### Bonus:

To pretty print a JSON submission file you can use: `python -m json.tool file.json`
