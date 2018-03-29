# -*- coding: utf-8 -*-
import json
from abc import ABC
from pathlib import Path
from typing import Tuple

from pandas import DataFrame

from evalutils.exceptions import FileLoaderError
from evalutils.io import first_int_in_filename_key, FileLoader
from evalutils.validators import DataFrameValidator


class Evaluation(ABC):
    def __init__(
        self,
        *,
        ground_truth_path: Path,
        predictions_path: Path = Path('/input/'),
        file_sorter_key=first_int_in_filename_key,
        file_loader: FileLoader,
        output_file: Path = Path('/output/metrics.json'),
        validators: Tuple[DataFrameValidator, ...] = (),
    ):
        super().__init__()
        self._ground_truth_path = ground_truth_path
        self._predictions_path = predictions_path
        self._file_sorter_key = file_sorter_key
        self._file_loader = file_loader
        self._output_file = output_file
        self._validators = validators

        self._ground_truth_cases = DataFrame()
        self._predictions_cases = DataFrame()

        self._case_results = DataFrame()
        self._aggregate_results = {}

    @property
    def _metrics(self):
        return {
            'case': self._case_results.to_dict(),
            'aggregates': self._aggregate_results,
        }

    def evaluate(self):
        self.load()
        self.validate()
        self.join_ground_truth_and_predictions()
        self.score()
        self.save()

    def load(self):
        self._ground_truth_cases = self._load_cases(
            folder=self._ground_truth_path
        )
        self._predictions_cases = self._load_cases(
            folder=self._predictions_path
        )

    def _load_cases(self, *, folder: Path) -> DataFrame:
        cases = DataFrame()
        for f in sorted(folder.glob('**/*'), key=self._file_sorter_key):
            try:
                cases.append(
                    self._file_loader.load(fname=f),
                    ignore_index=True,
                )
            except FileLoaderError:
                pass
        return cases

    def validate(self):
        self._validate_data_frame(df=self._ground_truth_cases)
        self._validate_data_frame(df=self._predictions_cases)
        self.cross_validate()

    def _validate_data_frame(self, *, df: DataFrame):
        for validator in self._validators:
            validator.validate(df=df)

    def cross_validate(self):
        pass

    def join_ground_truth_and_predictions(self):
        pass

    def score(self):
        pass

    def score_case(self):
        pass

    def score_aggregates(self):
        pass

    def save(self):
        pass

    def write_metrics_json(self):
        with open(self._output_file, 'w') as f:
            f.write(json.dumps(self._metrics))
