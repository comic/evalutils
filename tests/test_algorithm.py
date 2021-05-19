import json
import os
import shutil
from pathlib import Path
from typing import Dict

import SimpleITK
import numpy as np
from pandas import DataFrame
from scipy.ndimage import center_of_mass, label

from evalutils import (
    ClassificationAlgorithm,
    DetectionAlgorithm,
    SegmentationAlgorithm,
)


TEMPLATE_TEST_DIR = (
    Path(__file__).parent.parent
    / "evalutils"
    / "templates"
    / "algorithm"
    / "{{ cookiecutter.package_name }}"
    / "test"
)


class DetectionAlgorithmTest(DetectionAlgorithm):
    def predict(self, *, input_image: SimpleITK.Image) -> DataFrame:
        # Extract a numpy array with image data from the SimpleITK Image
        image_data = SimpleITK.GetArrayFromImage(input_image)

        # Detection: Compute connected components of the maximum values
        # in the input image and compute their center of mass
        sample_mask = image_data == np.max(image_data)
        labels, num_labels = label(sample_mask)
        candidates = center_of_mass(
            input=sample_mask, labels=labels, index=np.arange(num_labels) + 1
        )

        # Scoring: Score each candidate cluster with the value at its center
        candidate_scores = [
            image_data[tuple(coord)]
            for coord in np.array(candidates).astype(np.uint16)
        ]

        # Serialize candidates and scores as a list of dictionary entries
        data = self._serialize_candidates(
            candidates=candidates,
            candidate_scores=candidate_scores,
            ref_image=input_image,
        )

        # Convert serialized candidates to a pandas.DataFrame
        return DataFrame(data)


class SegmentationAlgorithmTest(SegmentationAlgorithm):
    def predict(self, *, input_image: SimpleITK.Image) -> SimpleITK.Image:
        # Segment all values greater than 2 in the input image
        return SimpleITK.BinaryThreshold(
            image1=input_image, lowerThreshold=2, insideValue=1, outsideValue=0
        )


class ClassificationAlgorithmTest(ClassificationAlgorithm):
    def predict(self, *, input_image: SimpleITK.Image) -> Dict:
        # Checks if there are any nodules voxels (> 1) in the input image
        return dict(
            values_exceeding_one=bool(
                np.any(SimpleITK.GetArrayFromImage(input_image) > 1)
            )
        )


def test_classification_algorithm(tmpdir):
    indir = Path(tmpdir / "input")
    shutil.copytree(TEMPLATE_TEST_DIR, indir)
    validate_algorithm_output(
        input_dir=indir,
        expected_results_file="results_classification.json",
        algorithm_test_class=ClassificationAlgorithmTest,
    )


def test_segmentation_algorithm(tmpdir):
    indir = Path(tmpdir / "input")
    out_file = Path(
        tmpdir
        / "output"
        / "images"
        / "1.0.000.000000.0.00.0.0000000000.0000.0000000000.000.mhd"
    )
    shutil.copytree(TEMPLATE_TEST_DIR, indir)
    validate_algorithm_output(
        input_dir=indir,
        expected_results_file="results_segmentation.json",
        algorithm_test_class=SegmentationAlgorithmTest,
    )
    assert out_file.exists()
    out_img = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(str(out_file)))
    in_img = SimpleITK.GetArrayFromImage(
        SimpleITK.ReadImage(str(indir / out_file.name))
    )
    assert np.array_equal((in_img >= 2), (out_img > 0))


def test_detection_algorithm(tmpdir):
    indir = tmpdir / "input"
    shutil.copytree(TEMPLATE_TEST_DIR, indir)
    validate_algorithm_output(
        input_dir=indir,
        expected_results_file="results_detection.json",
        algorithm_test_class=DetectionAlgorithmTest,
    )


def test_detection_algorithm_2d_input(tmpdir):
    indir = tmpdir / "input"

    os.makedirs(indir)
    test_image = (
        TEMPLATE_TEST_DIR
        / "1.0.000.000000.0.00.0.0000000000.0000.0000000000.000.mhd"
    )
    image_data = SimpleITK.GetArrayFromImage(
        SimpleITK.ReadImage(str(test_image))
    )[74, :, :]
    SimpleITK.WriteImage(
        SimpleITK.GetImageFromArray(image_data),
        str(indir / "2dtest.mha"),
        True,
    )

    validate_algorithm_output(
        input_dir=indir,
        expected_results_file="results_2d.json",
        algorithm_test_class=DetectionAlgorithmTest,
    )


def test_detection_algorithm_empty_input(tmpdir):
    indir = tmpdir / "input"

    os.makedirs(indir)
    SimpleITK.WriteImage(
        SimpleITK.GetImageFromArray(np.zeros((100, 100), dtype=np.uint8)),
        str(indir / "emptytest.mha"),
        True,
    )

    validate_algorithm_output(
        input_dir=indir,
        expected_results_file="results_empty.json",
        algorithm_test_class=DetectionAlgorithmTest,
    )


def validate_algorithm_output(
    input_dir: Path, expected_results_file: str, algorithm_test_class: type
):
    output_dir = Path(input_dir).parent / "output"
    output_dir.mkdir()
    proc = algorithm_test_class()
    proc._input_path = Path(input_dir)
    proc._output_file = Path(output_dir) / "results.json"
    proc._output_path = Path(output_dir) / "images"
    proc.process()
    results_file = output_dir / "results.json"
    assert results_file.exists()
    with open(str(results_file)) as f:
        results = json.load(f)

    expected_path = (
        Path(__file__).parent / "resources" / "json" / expected_results_file
    )
    if not expected_path.exists():
        expected_path = TEMPLATE_TEST_DIR / expected_results_file

    with open(str(expected_path)) as f:
        expected_result = json.load(f)
    assert results == expected_result
