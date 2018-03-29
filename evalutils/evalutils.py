# -*- coding: utf-8 -*-
import json
from abc import ABC
from pathlib import Path
from typing import Tuple

from pandas import DataFrame, merge

from evalutils.exceptions import FileLoaderError, ValidationError
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
        validators: Tuple[DataFrameValidator, ...] = (),
        join_column: str = None,
        output_file: Path = Path('/output/metrics.json'),
    ):
        super().__init__()
        self._ground_truth_path = ground_truth_path
        self._predictions_path = predictions_path
        self._file_sorter_key = file_sorter_key
        self._file_loader = file_loader
        self._join_column = join_column
        self._validators = validators
        self._output_file = output_file

        self._ground_truth_cases = DataFrame()
        self._predictions_cases = DataFrame()

        self._cases = DataFrame()

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
        self.merge_ground_truth_and_predictions()
        self.cross_validate()
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
                # Couldn't load this file with this loader, but don't worry
                # about it. This will be found in the validation.
                pass
        return cases

    def validate(self):
        self._validate_data_frame(df=self._ground_truth_cases)
        self._validate_data_frame(df=self._predictions_cases)

    def _validate_data_frame(self, *, df: DataFrame):
        for validator in self._validators:
            validator.validate(df=df)

    def merge_ground_truth_and_predictions(self):
        if self._join_column:
            kwargs = {'on': self._join_column}
        else:
            kwargs = {'left_index': True, 'right_index': True}

        self._cases = merge(
            left=self._ground_truth_cases,
            right=self._predictions_cases,
            indicator=True,
            how='outer',
            suffixes=('ground_truth', 'prediction'),
            **kwargs,
        )

    def cross_validate(self):
        missing = [p for _, p in self._cases.iterrows() if
                   p['_merge'] == 'left_only']
        extra = [p for _, p in self._cases.iterrows() if
                 p['_merge'] == 'right_only']

        if missing:
            self._raise_missing_predictions_error(missing=missing)

        if extra:
            self._raise_extra_predictions_error(extra=extra)

    def _raise_missing_predictions_error(self, *, missing):
        if self._join_column:
            missing = [p[self._join_column] for p in missing]
            message = (
                'Predictions missing: you did not submit predictions for '
                f'{self._join_column}: {missing}. Please try again.'
            )
        else:
            message = (
                'Predictions missing: you did not submit enough predictions,'
                'please try again.'
            )

        raise ValidationError(message)

    def _raise_extra_predictions_error(self, *, extra):
        if self._join_column:
            extra = [p[self._join_column] for p in extra]
            message = (
                'Too many predictions: we do not have the ground truth data '
                f'for {self._join_column}: {extra}. Please try again.'
            )
        else:
            message = (
                'Too many predictions: you submitted too many predictions, '
                'please try again.'
            )

        raise ValidationError(message)

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
