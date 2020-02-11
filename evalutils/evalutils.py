import json
import logging
from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import Tuple, Dict, Set, Callable, List, Union, Pattern
from warnings import warn

from pandas import DataFrame, merge, Series, concat

from .exceptions import FileLoaderError, ValidationError, ConfigurationError
from .io import first_int_in_filename_key, FileLoader, CSVLoader
from .scorers import score_detection
from .validators import DataFrameValidator

logger = logging.getLogger(__name__)


class BaseAlgorithm(ABC):
    def __init__(
        self,
        *,
        index_key: str,
        file_loaders: Dict[str, FileLoader],
        file_filters: Dict[str, Pattern[str]] = None,
        input_path: Path = Path("/input/"),
        output_path: Path = Path("/output/images/"),
        file_sorter_key: Callable = None,
        validators: Dict[str, Tuple[DataFrameValidator, ...]],
        output_file: PathLike = Path("/output/results.json"),
    ):
        """
        The base class for all algorithms. Sets the environment and controls
        the flow of the processing once `process` is called.


        Parameters
        ----------
        index_key
            Fileloader key which must be used for the index
        file_loaders
            The loaders that will be used to get all files
        file_filters
            Regular expressions for filtering certain FileLoaders
        input_path
            The path in the container where the ground truth will be loaded
            from
        output_path
            The path in the container where the output images will be written
        file_sorter_keys
            A function that determines how files in the input_path are sorted
        validators
            A dictionary containing the validators that will be used on the
            loaded data per file_loader key
        output_file
            The path to the location where the results will be written
        """
        self._index_key = index_key
        self._input_path = input_path
        self._output_path = output_path
        self._file_sorter_key = file_sorter_key
        self._file_filters = file_filters
        self._file_loaders = file_loaders
        self._validators = validators
        self._output_file = output_file

        self._ground_truth_cases = DataFrame()
        self._predictions_cases = DataFrame()

        self._cases = {}

        self._case_results = []

        super().__init__()

    def load(self):
        for key, file_loader in self._file_loaders.items():
            filter = (
                self._file_filters[key] if key in self._file_filters else None
            )
            self._cases[key] = self._load_cases(
                folder=self._input_path, file_loader=file_loader, filter=filter
            )

    def _load_cases(
        self,
        *,
        folder: Path,
        file_loader: FileLoader,
        filter: Pattern[str] = None,
    ) -> DataFrame:
        cases = None

        for f in sorted(folder.glob("**/*"), key=self._file_sorter_key):
            if filter is None or filter.match(str(f)):
                try:
                    new_cases = file_loader.load(fname=f)
                except FileLoaderError:
                    logger.warning(
                        f"Could not load {f.name} using {file_loader}."
                    )
                else:
                    if cases is None:
                        cases = new_cases
                    else:
                        cases += new_cases
            else:
                logger.info(
                    f"Skip loading {f.name} because it doesn't match {filter}."
                )

        if cases is None:
            raise FileLoaderError(
                f"Could not load any files in {folder} with " f"{file_loader}."
            )

        return DataFrame(cases)

    def validate(self):
        """ Validates each dataframe for each fileloader separately """
        file_loaders_keys = [k for k in self._file_loaders.keys()]
        for key in self._validators.keys():
            if key not in file_loaders_keys:
                raise ValueError(
                    f"There is no file_loader associated with: {key}.\n"
                    f"Valid file loaders are: {file_loaders_keys}"
                )
        for key, cases in self._cases.items():
            if key in self._validators:
                self._validate_data_frame(df=cases, file_loader_key=key)

    def _validate_data_frame(self, *, df: DataFrame, file_loader_key: str):
        for validator in self._validators[file_loader_key]:
            validator.validate(df=df)

    def process(self):
        self.load()
        self.validate()
        self.process_cases()
        self.save()

    def process_cases(self, file_loader_key: str = None):
        if file_loader_key is None:
            file_loader_key = self._index_key
        self._case_results = []
        for idx, case in self._cases[file_loader_key].iterrows():
            self._case_results.append(self.process_case(idx=idx, case=case))

    # noinspection PyUnusedLocal
    def process_case(self, *, idx: int, case: DataFrame) -> Dict:
        return {}

    def save(self):
        with open(self._output_file, "w") as f:
            json.dump(self._case_results, f)


class BaseEvaluation(ABC):
    def __init__(
        self,
        *,
        ground_truth_path: Path = Path("/opt/evaluation/ground-truth/"),
        predictions_path: Path = Path("/input/"),
        file_sorter_key: Callable = first_int_in_filename_key,
        file_loader: FileLoader,
        validators: Tuple[DataFrameValidator, ...],
        join_key: str = None,
        aggregates: Set[str] = None,
        output_file: PathLike = Path("/output/metrics.json"),
    ):
        """
        The base class for all evaluations. Sets the environment and controls
        the flow of the evaluation once `evaluate` is called.


        Parameters
        ----------
        ground_truth_path
            The path in the container where the ground truth will be loaded
            from
        predictions_path
            The path in the container where the submission will be loaded from
        file_sorter_key
            A function that determines how files are sorted and matched
            together
        file_loader
            The loader that will be used to get all files
        validators
            A tuple containing all the validators that will be used on the
            loaded data
        join_key
            The column that will be used to join the predictions and ground
            truth tables
        aggregates
            The set of aggregates that will be calculated by
            `pandas.DataFrame.describe`
        output_file
            The path to the location where the results will be written
        """
        if aggregates is None:
            aggregates = {
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
            }

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
    def _metrics(self) -> Dict:
        """ Returns the calculated case and aggregate results """
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
                    cases = new_cases
                else:
                    cases += new_cases

        if cases is None:
            raise FileLoaderError(
                f"Could not load any files in {folder} with "
                f"{self._file_loader}."
            )

        return DataFrame(cases)

    def validate(self):
        """ Validates each dataframe separately """
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
        """ Validates both dataframes """
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
    def score_case(self, *, idx: int, case: DataFrame) -> Dict:
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
        missing = [
            p for _, p in self._cases.iterrows() if p["_merge"] == "left_only"
        ]

        if missing:
            if self._join_key:
                missing = [p[self._join_key] for p in missing]
            self._raise_missing_predictions_error(missing=missing)

        extra = [
            p for _, p in self._cases.iterrows() if p["_merge"] == "right_only"
        ]

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
            DeprecationWarning,
        )
        super().__init__(*args, **kwargs)


class DetectionEvaluation(BaseEvaluation):
    """
    DetectionEvaluations have a different number of predictions from the
    number of ground truth annotations. An example would be detecting lung
    nodules in a CT volume, or malignant cells in a pathology slide.
    """

    def __init__(self, *args, detection_radius, detection_threshold, **kwargs):
        super().__init__(*args, **kwargs)
        self._detection_radius = detection_radius
        self._detection_threshold = detection_threshold

    def merge_ground_truth_and_predictions(self):
        self._cases = concat(
            [self._ground_truth_cases, self._predictions_cases],
            keys=["ground_truth", "predictions"],
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

    def _raise_extra_predictions_error(self, *, extra=None):
        """ In detection challenges extra predictions are ok """
        warn(f"There are extra predictions for cases: {extra}.")

    def _raise_missing_predictions_error(self, *, missing=None):
        """ In detection challenges missing predictions are ok """
        warn(f"Could not find predictions for cases: {missing}.")

    def score(self):
        cases = set(self._ground_truth_cases[self._join_key])
        cases |= set(self._predictions_cases[self._join_key])

        self._case_results = DataFrame()

        for idx, case in enumerate(cases):
            self._case_results = self._case_results.append(
                self.score_case(
                    idx=idx,
                    case=self._cases.loc[self._cases[self._join_key] == case],
                ),
                ignore_index=True,
            )
        self._aggregate_results = self.score_aggregates()

    def score_case(self, *, idx, case):
        score = score_detection(
            ground_truth=self.get_points(case=case, key="ground_truth"),
            predictions=self.get_points(case=case, key="predictions"),
            radius=self._detection_radius,
        )

        # Add the case id to the score
        output = score._asdict()
        output.update({self._join_key: case[self._join_key][0]})

        return output

    def get_points(
        self, *, case, key: str
    ) -> List[Tuple[Union[int, float], Union[int, float]]]:
        raise NotImplementedError

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
