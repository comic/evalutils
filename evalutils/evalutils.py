# -*- coding: utf-8 -*-
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Dict, Set, Callable
from warnings import warn

from pandas import DataFrame, merge, Series, concat

from .exceptions import FileLoaderError, ValidationError, ConfigurationError
from .io import first_int_in_filename_key, FileLoader, CSVLoader
from .validators import DataFrameValidator

logger = logging.getLogger(__name__)


class BaseEvaluation(ABC):
    def __init__(
        self,
        *,
        ground_truth_path: Path = Path("/usr/src/evaluation/ground-truth/"),
        predictions_path: Path = Path("/input/"),
        file_sorter_key: Callable = first_int_in_filename_key,
        file_loader: FileLoader,
        validators: Tuple[DataFrameValidator, ...],
        join_key: str = None,
        aggregates: Set[str] = {
            "mean",
            "std",
            "min",
            "max",
            "25%",
            "50%",
            "75%",
            "count",
            "uniq",
            "freq",
        },
        output_file: Path = Path("/output/metrics.json"),
    ):
        self._ground_truth_path = ground_truth_path
        self._predictions_path = predictions_path
        self._file_sorter_key = file_sorter_key
        self._file_loader = file_loader
        self._validators = validators
        self._join_key = join_key
        self._aggregates = aggregates
        self._output_file = output_file

        self._ground_truth_cases = DataFrame()
        self._predictions_cases = DataFrame()

        self._cases = DataFrame()

        self._case_results = DataFrame()
        self._aggregate_results = {}
        super().__init__()

        if isinstance(self._file_loader, CSVLoader) and self._join_key is None:
            raise ConfigurationError(
                f"You must set a `join_key` when using {self._file_loader}."
            )

    @property
    def _metrics(self):
        return {
            "case": self._case_results.to_dict(),
            "aggregates": self._aggregate_results,
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
        cases = None

        for f in sorted(folder.glob("**/*"), key=self._file_sorter_key):
            try:
                new_cases = self._file_loader.load(fname=f)
            except FileLoaderError:
                logger.warning(
                    f"Could not load {f.name} using {self._file_loader}."
                )
            else:
                if cases is None:
                    cases = DataFrame(new_cases)
                else:
                    cases = cases.append(new_cases, ignore_index=True)

        if cases is None:
            raise FileLoaderError(
                f"Could not load and files in {folder} with "
                f"{self._file_loader}."
            )

        return cases

    def validate(self):
        self._validate_data_frame(df=self._ground_truth_cases)
        self._validate_data_frame(df=self._predictions_cases)

    def _validate_data_frame(self, *, df: DataFrame):
        for validator in self._validators:
            validator.validate(df=df)

    @abstractmethod
    def merge_ground_truth_and_predictions(self):
        pass

    @abstractmethod
    def cross_validate(self):
        pass

    def _raise_missing_predictions_error(self, *, missing=None):
        if missing is not None:
            message = (
                "Predictions missing: you did not submit predictions for "
                f"{missing}. Please try again."
            )
        else:
            message = (
                "Predictions missing: you did not submit enough predictions, "
                "please try again."
            )

        raise ValidationError(message)

    def _raise_extra_predictions_error(self, *, extra=None):
        if extra is not None:
            message = (
                "Too many predictions: we do not have the ground truth data "
                f"for {extra}. Please try again."
            )
        else:
            message = (
                "Too many predictions: you submitted too many predictions, "
                "please try again."
            )

        raise ValidationError(message)

    @abstractmethod
    def score(self):
        pass

    # noinspection PyUnusedLocal
    @staticmethod
    def score_case(*, idx: int, case: DataFrame) -> Dict:
        return {}

    def score_aggregates(self) -> Dict:
        aggregate_results = {}

        for col in self._case_results.columns:
            aggregate_results[col] = self.aggregate_series(
                series=self._case_results[col]
            )

        return aggregate_results

    def aggregate_series(self, *, series: Series) -> Dict:
        summary = series.describe()
        valid_keys = [a for a in self._aggregates if a in summary]

        series_summary = {}

        for k in valid_keys:
            value = summary[k]

            # % in keys could cause problems when looking up values later
            key = k.replace("%", "pc")

            try:
                json.dumps(value)
            except TypeError:
                logger.warning(
                    f"Could not serialize {key}: {value} as json, "
                    f"so converting {value} to int."
                )
                value = int(value)

            series_summary[key] = value

        return series_summary

    def save(self):
        self.write_metrics_json()

    def write_metrics_json(self):
        with open(self._output_file, "w") as f:
            f.write(json.dumps(self._metrics))


class ClassificationEvaluation(BaseEvaluation):
    """
    ClassificationEvaluations have the same number of predictions as the
    number of ground truth cases. These can be things like, what is the
    stage of this case, or segment some things in this case.
    """

    def merge_ground_truth_and_predictions(self):
        if self._join_key:
            kwargs = {"on": self._join_key}
        else:
            kwargs = {"left_index": True, "right_index": True}

        self._cases = merge(
            left=self._ground_truth_cases,
            right=self._predictions_cases,
            indicator=True,
            how="outer",
            suffixes=("_ground_truth", "_prediction"),
            **kwargs,
        )

    def cross_validate(self):
        missing = [p for _, p in self._cases.iterrows() if
                   p["_merge"] == "left_only"]

        if missing:
            if self._join_key:
                missing = [p[self._join_key] for p in missing]
            self._raise_missing_predictions_error(missing=missing)

        extra = [p for _, p in self._cases.iterrows() if
                 p["_merge"] == "right_only"]

        if extra:
            if self._join_key:
                extra = [p[self._join_key] for p in extra]
            self._raise_extra_predictions_error(extra=extra)

    def score(self):
        self._case_results = DataFrame()
        for idx, case in self._cases.iterrows():
            self._case_results = self._case_results.append(
                self.score_case(idx=idx, case=case), ignore_index=True
            )
        self._aggregate_results = self.score_aggregates()


class Evaluation(ClassificationEvaluation):
    """
    Legacy class, you should use ClassificationEvaluation instead.
    """

    def __init__(self, *args, **kwargs):
        warn(
            (
                "The Evaluation class is deprecated, "
                "please use ClassificationEvaluation instead"
            ),
            DeprecationWarning
        )
        super().__init__(*args, **kwargs)


class DetectionEvaluation(BaseEvaluation):
    """
    DetectionEvaluations have a different number of predictions from the
    number of ground truth annotations. An example would be detecting lung
    nodules in a CT volume, or malignant cells in a pathology slide.
    """

    def merge_ground_truth_and_predictions(self):
        self._cases = concat(
            [self._ground_truth_cases, self._predictions_cases],
            keys=["ground_truth", "predictions"]
        )

    def cross_validate(self):
        expected_keys = set(self._ground_truth_cases[self._join_key])
        submitted_keys = set(self._predictions_cases[self._join_key])

        missing = expected_keys - submitted_keys
        if missing:
            self._raise_missing_predictions_error(missing=missing)

        extra = submitted_keys - expected_keys
        if extra:
            self._raise_extra_predictions_error(extra=extra)

    def score(self):
        cases = set(self._ground_truth_cases[self._join_key])

        self._case_results = DataFrame()

        for idx, case in enumerate(cases):
            self._case_results = self._case_results.append(
                self.score_case(
                    idx=idx,
                    case=self._cases.loc[self._cases[self._join_key] == case],
                ), ignore_index=True
            )
        self._aggregate_results = self.score_aggregates()

    def score_aggregates(self):
        aggregate_results = super().score_aggregates()

        totals = self._case_results.sum()

        for s in totals.index:
            aggregate_results[s]["sum"] = totals[s]

        tp = aggregate_results["true_positives"]["sum"]
        fp = aggregate_results["false_positives"]["sum"]
        fn = aggregate_results["false_negatives"]["sum"]

        aggregate_results["precision"] = tp / (tp + fp)
        aggregate_results["recall"] = tp / (tp + fn)
        aggregate_results["f1_score"] = 2 * tp / ((2 * tp) + fp + fn)

        return aggregate_results
