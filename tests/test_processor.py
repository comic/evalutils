from pathlib import Path
import SimpleITK
import shutil
from pandas import DataFrame
import re
import os
import json
from evalutils import BaseProcess
from evalutils.io import SimpleITKLoader, CSVLoader
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
    ExpectedColumnNamesValidator,
)


class BasicProcessTest(BaseProcess):
    def __init__(self, outdir, input_path):
        super().__init__(
            index_key="lung",
            file_loaders=dict(lung=SimpleITKLoader(), nodules=CSVLoader()),
            file_filters=dict(
                lung=re.compile(r"^.*\.mhd$"), nodules=re.compile(r"^.*\.csv$")
            ),
            validators=dict(
                lung=(UniqueImagesValidator(), UniquePathIndicesValidator()),
                nodules=(
                    ExpectedColumnNamesValidator(
                        expected=(
                            "seriesuid",
                            "coordX",
                            "coordY",
                            "coordZ",
                            "class",
                        )
                    ),
                ),
            ),
            input_path=Path(input_path),
            output_file=Path(outdir) / "results.json",
        )
        self._scored_nodules = DataFrame()

    def process_case(self, *, idx, case):
        lung_path = case["path"]

        all_nodules = self._cases["nodules"]
        are_nodules_for_case = (
            all_nodules["seriesuid"] == lung_path.name.rsplit(".", 1)[0]
        )
        nodules = all_nodules[are_nodules_for_case].copy()

        # Load the images and annotations for this case
        lung = self._file_loaders["lung"].load_image(lung_path)

        # Check that they're the expected images and annotations
        assert self._file_loaders["lung"].hash_image(lung) == case["hash"]
        lung = SimpleITK.GetArrayFromImage(lung)

        scored_nodules = self.predict(lung, nodules)

        self._scored_nodules.append(scored_nodules)

        return {
            "scored_nodules": scored_nodules.to_dict(),
            "lung_fname": lung_path.name,
        }

    def predict(self, lung_image, nodules_locations):
        scores = []
        for nodule in nodules_locations.iterrows():
            scores.append(0.5)
        nodules_locations.loc[:, "class"] = scores
        return nodules_locations


def test_detection_evaluation(tmpdir):
    indir = tmpdir / "input"
    outdir = tmpdir / "output"
    resdir = Path(__file__).parent / "resources"

    os.makedirs(outdir)
    shutil.copytree(resdir / "itk", indir)
    shutil.copy(
        resdir / "nodules" / "candidates_001.csv", indir / "candidates_001.csv"
    )

    proc = BasicProcessTest(input_path=indir, outdir=outdir)

    proc.process()

    results_file = outdir / "results.json"

    assert results_file.exists()

    with open(results_file, "r") as f:
        results = json.load(f)

    print(results)
