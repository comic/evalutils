#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path

from evalutils import Evaluation
from evalutils.io import CSVLoader
from evalutils.validators import ExpectedColumnNamesValidator


class TestEval(Evaluation):
    def __init__(self):
        super().__init__(
            file_loader=CSVLoader(),
            ground_truth_path=(
                Path(__file__).parent / 'resources' / 'reference'
            ),
            predictions_path=(
                Path(__file__).parent / 'resources' / 'submission'
            ),
            output_file=Path('/tmp/metrics.json'),
            join_key='case',
            validators=(
                ExpectedColumnNamesValidator(expected=('case', 'class',)),
            ),
        )

def test_class_creation():
    TestEval().evaluate()
