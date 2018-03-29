#!/usr/bin/env python
# -*- coding: utf-8 -*-
from evalutils import Evaluation


def test_class_creation():
    # noinspection PyUnusedLocal
    class TestEval(Evaluation):
        def __init__(self):
            super().__init__(
                file_loader=None,
                ground_truth_path=None,
            )

    e = TestEval()

