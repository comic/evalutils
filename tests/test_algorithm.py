import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import SimpleITK
import numpy as np
from pandas import DataFrame

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
        assert self._file_loaders["lung"].hash_image(lung) == case["hash"]

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

        # Detection: Select a maximum of 10 candidates (image coordinate)
        # using a fixed random seed
        sample_mask = image_data > 0
        candidates = self._sample_fixed_random_candidates(
            sample_mask=sample_mask, num_samples=10, random_seed_value=42
        )

        # Scoring: Score each candidate with the value at its image coordinate
        candidate_scores = [image_data[coord] for coord in candidates]

        # Serialize candidates as a list of dictionary entries
        data = self._serialize_candidates(
            candidates=candidates,
            candidate_scores=candidate_scores,
            ref_image=input_image,
        )

        # Convert serialized candidates to a DataFrame
        return DataFrame(data)

    def _serialize_candidates(
        self,
        *,
        candidates: List[Tuple[np.int, ...]],
        candidate_scores: List[Any],
        ref_image: SimpleITK.Image,
    ) -> List[Dict]:
        data = []
        for coord, score in zip(candidates, candidate_scores):
            world_coords = ref_image.TransformIndexToPhysicalPoint(
                [int(c) for c in reversed(coord)]
            )
            coord_data = {
                f"coord{k}": v for k, v in zip(["X", "Y", "Z"], world_coords)
            }
            coord_data.update({"score": score})
            data.append(coord_data)
        return data

    def _sample_fixed_random_candidates(
        self,
        *,
        sample_mask: np.array,
        num_samples: int = 10,
        random_seed_value: int = 42,
    ) -> List[Tuple[np.int, ...]]:
        candidates = [e for e in zip(*np.where(sample_mask))]
        if len(candidates) > 0:
            np.random.seed(seed=random_seed_value)
            indices = np.random.choice(
                len(candidates),
                min(num_samples, len(candidates)),
                replace=False,
            )
            candidates = [candidates[idx] for idx in indices]
        return candidates


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
    check_algorithm_output(
        input_dir=indir, expected_results_file="results.json"
    )


def test_detection_algorithm_2d_input(tmpdir):
    indir = tmpdir / "input"

    os.makedirs(indir)
    test_image = Path(__file__).parent / "resources" / "images" / "1_mask.png"
    SimpleITK.WriteImage(
        SimpleITK.ReadImage(str(test_image)), str(indir / "2dtest.mha"), True
    )

    check_algorithm_output(
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

    check_algorithm_output(
        input_dir=indir, expected_results_file="results_empty.json"
    )


def check_algorithm_output(input_dir: Path, expected_results_file: str):
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
