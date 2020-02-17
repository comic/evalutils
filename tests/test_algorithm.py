import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import SimpleITK
import numpy as np
from pandas import DataFrame
from scipy.ndimage import center_of_mass, label

from evalutils import BaseAlgorithm
from evalutils.io import SimpleITKLoader
from evalutils.validators import (
    UniqueImagesValidator,
    UniquePathIndicesValidator,
)


TEMPLATE_TEST_DIR = (
    Path(__file__).parent.parent
    / "evalutils"
    / "templates"
    / "algorithm"
    / "{{ cookiecutter.package_name }}"
    / "test"
)


class BasicAlgorithmTest(BaseAlgorithm):
    def __init__(self, outdir: Path, input_path: Path):
        super().__init__(
            index_key="input_image",
            file_loaders=dict(input_image=SimpleITKLoader()),
            file_filters=dict(input_image=re.compile(r"^.*\.mh[ad]$")),
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path=Path(input_path),
            output_file=Path(outdir) / "results.json",
            output_path=Path(outdir) / "images",
        )


class DetectionAlgorithmTest(BasicAlgorithmTest):
    def process_case(self, *, idx, case):
        input_image_file_path = case["path"]

        # Load the image for this case
        input_image = self._file_loaders["input_image"].load_image(
            input_image_file_path
        )

        # Check that it is the expected image
        if (
            self._file_loaders["input_image"].hash_image(input_image)
            != case["hash"]
        ):
            raise RuntimeError("Image hashes do not match")

        # Detect and score candidates
        scored_candidates = self.predict(input_image=input_image)

        # Write resulting candidates to result.json for this case
        return {
            "outputs": [
                dict(type="candidates", data=scored_candidates.to_dict())
            ],
            "inputs": [
                dict(type="metaio_image", filename=input_image_file_path.name)
            ],
            "error_messages": [],
        }

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

    def _serialize_candidates(
        self,
        *,
        candidates: Iterable[Tuple[float, ...]],
        candidate_scores: List[Any],
        ref_image: SimpleITK.Image,
    ) -> List[Dict]:
        data = []
        for coord, score in zip(candidates, candidate_scores):
            world_coords = ref_image.TransformContinuousIndexToPhysicalPoint(
                [c for c in reversed(coord)]
            )
            coord_data = {
                f"coord{k}": v for k, v in zip(["X", "Y", "Z"], world_coords)
            }
            coord_data.update({"score": score})
            data.append(coord_data)
        return data


class SegmentationAlgorithmTest(BasicAlgorithmTest):
    def process_case(self, *, idx, case):
        input_image_file_path = case["path"]

        # Load the image for this case
        input_image = self._file_loaders["input_image"].load_image(
            input_image_file_path
        )

        # Check that it is the expected image
        if (
            self._file_loaders["input_image"].hash_image(input_image)
            != case["hash"]
        ):
            raise RuntimeError("Image hashes do not match")

        # Segment nodule candidates
        segmented_nodules = self.predict(input_image=input_image)

        # Write resulting segmentation to output location
        segmentation_path = self._output_path / input_image_file_path.name
        if not self._output_path.exists():
            self._output_path.mkdir()
        SimpleITK.WriteImage(segmented_nodules, str(segmentation_path), True)

        # Write segmentation file path to result.json for this case
        return {
            "outputs": [
                dict(type="metaio_image", filename=segmentation_path.name)
            ],
            "inputs": [
                dict(type="metaio_image", filename=input_image_file_path.name)
            ],
            "error_messages": [],
        }

    def predict(self, *, input_image: SimpleITK.Image) -> SimpleITK.Image:
        # Segment all values greater than 2 in the input image
        return SimpleITK.BinaryThreshold(
            image1=input_image, lowerThreshold=2, insideValue=1, outsideValue=0
        )


class ClassificationAlgorithmTest(BasicAlgorithmTest):
    def process_case(self, *, idx, case):
        input_image_file_path = case["path"]

        # Load the image for this case
        input_image = self._file_loaders["input_image"].load_image(
            input_image_file_path
        )

        # Check that it is the expected image
        if (
            self._file_loaders["input_image"].hash_image(input_image)
            != case["hash"]
        ):
            raise RuntimeError("Image hashes do not match")

        # Classify input_image image
        values_exceeding_one = self.predict(input_image=input_image)

        # Write resulting classification to result.json for this case
        return {
            "outputs": [
                dict(
                    type="bool",
                    name="values_exceeding_one",
                    value=values_exceeding_one,
                )
            ],
            "inputs": [
                dict(type="metaio_image", filename=input_image_file_path.name)
            ],
            "error_messages": [],
        }

    def predict(self, *, input_image: SimpleITK.Image) -> bool:
        # Checks if there are any nodules voxels (> 1) in the input image
        return bool(np.any(SimpleITK.GetArrayFromImage(input_image) > 1))


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
    proc = algorithm_test_class(input_path=input_dir, outdir=output_dir)
    proc.process()
    results_file = output_dir / "results.json"
    assert results_file.exists()
    with open(str(results_file), "r") as f:
        results = json.load(f)
    print(results)
    expected_path = (
        Path(__file__).parent / "resources" / "json" / expected_results_file
    )
    with open(str(expected_path), "r") as f:
        expected_result = json.load(f)
    assert results == expected_result
