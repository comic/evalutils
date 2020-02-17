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


class BasicAlgorithmTest(BaseAlgorithm):
    def __init__(self, outdir, input_path):
        super().__init__(
            index_key="lung",
            file_loaders=dict(lung=SimpleITKLoader()),
            file_filters=dict(lung=re.compile(r"^.*\.mh[ad]$")),
            validators=dict(
                lung=(UniqueImagesValidator(), UniquePathIndicesValidator())
            ),
            input_path=Path(input_path),
            output_file=Path(outdir) / "results.json",
        )

    def process_case(self, *, idx, case):
        lung_path = case["path"]

        # Load the image for this case
        lung = self._file_loaders["lung"].load_image(lung_path)

        # Check that it is the expected image
        if self._file_loaders["lung"].hash_image(lung) != case["hash"]:
            raise RuntimeError("Image hashes do not match")

        # Detect and score candidates
        scored_candidates = self.predict(input_image=lung)

        # Write resulting candidates to result.json for this case
        return {
            "outputs": [
                dict(type="candidates", data=scored_candidates.to_dict())
            ],
            "inputs": [dict(type="metaio_image", filename=lung_path.name)],
            "error_messages": [],
        }

    def predict(self, *, input_image: SimpleITK.Image) -> DataFrame:
        # Extract a numpy array with image data from the SimpleITK Image
        image_data = SimpleITK.GetArrayFromImage(input_image)

        # Detection: Compute connected components of all values greater than 2
        # in the input image and compute their center of mass
        sample_mask = image_data >= 2
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


def test_detection_algorithm(tmpdir):
    indir = tmpdir / "input"
    resdir = (
        Path(__file__).parent.parent
        / "evalutils"
        / "templates"
        / "algorithm"
        / "{{ cookiecutter.package_name }}"
        / "test"
    )
    shutil.copytree(resdir, indir)
    validate_algorithm_output(
        input_dir=indir, expected_results_file="results.json"
    )


def test_detection_algorithm_2d_input(tmpdir):
    indir = tmpdir / "input"

    os.makedirs(indir)
    test_image = (
        Path(__file__).parent.parent
        / "evalutils"
        / "templates"
        / "algorithm"
        / "{{ cookiecutter.package_name }}"
        / "test"
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
        input_dir=indir, expected_results_file="results_2d.json"
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
        input_dir=indir, expected_results_file="results_empty.json"
    )


def validate_algorithm_output(input_dir: Path, expected_results_file: str):
    output_dir = Path(input_dir).parent / "output"
    output_dir.mkdir()
    proc = BasicAlgorithmTest(input_path=input_dir, outdir=output_dir)
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
