import json
import os
import re
import shutil
from pathlib import Path

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
        self._scored_nodules = DataFrame()

    def process_case(self, *, idx, case):
        lung_path = case["path"]

        # Load the images for this case
        lung = self._file_loaders["lung"].load_image(lung_path)

        # Check that they're the expected images
        assert self._file_loaders["lung"].hash_image(lung) == case["hash"]

        scored_nodules = self.predict(lung)

        self._scored_nodules.append(scored_nodules)

        return {
            "outputs": [dict(data=scored_nodules.to_dict(), type="nodules")],
            "inputs": [dict(type="metaio_image", filename=lung_path.name)],
            "error_messages": [],
        }

    def predict(self, lung_image: SimpleITK.Image) -> DataFrame:
        lung_data = SimpleITK.GetArrayFromImage(lung_image)

        # Take 10 random points with a fixed seed
        np.random.seed(42)
        candidates = [(z, y, x) for z, y, x in zip(*np.where(lung_data > 0))]
        indices = np.random.choice(len(candidates), 10)
        candidates = [candidates[idx] for idx in indices]

        return DataFrame(
            [
                {
                    "coordX": x_world,
                    "coordY": y_world,
                    "coordZ": z_world,
                    "class": lung_data[z, y, x],
                }
                for z, y, x in candidates
                for x_world, y_world, z_world in [
                    lung_image.TransformIndexToPhysicalPoint(
                        (int(x), int(y), int(z))
                    )
                ]
            ]
        )


def test_nodule_detection_algorithm(tmpdir):
    indir = tmpdir / "input"
    outdir = tmpdir / "output"

    resdir = (
        Path(__file__).parent.parent
        / "evalutils"
        / "templates"
        / "algorithm"
        / "{{ cookiecutter.package_name }}"
        / "test"
    )

    os.makedirs(outdir)
    shutil.copytree(resdir, indir)

    proc = BasicAlgorithmTest(input_path=indir, outdir=outdir)

    proc.process()

    results_file = outdir / "results.json"

    assert results_file.exists()

    with open(results_file, "r") as f:
        results = json.load(f)

    print(results)

    expected_results_file = (
        Path(__file__).parent / "resources" / "json" / "results.json"
    )

    with open(expected_results_file, "r") as f:
        expected_result = json.load(f)

    assert results == expected_result
