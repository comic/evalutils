from pathlib import Path
import SimpleITK
import shutil
from pandas import DataFrame
import re
import os
import json
from evalutils import BaseAlgorithm
from evalutils.io import SimpleITKLoader, CSVLoader
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
    ExpectedColumnNamesValidator,
)


class BasicAlgorithmTest(BaseAlgorithm):
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
        self._nodule_files = self._input_path.glob("*.csv")
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

        scored_nodules = self.predict(lung, nodules)

        self._scored_nodules.append(scored_nodules)

        return {
            "outputs": [
                dict(
                    data=scored_nodules[scored_nodules.keys()[1:]].to_dict(),
                    type="nodules",
                )
            ],
            "inputs": [
                dict(type="metaio_image", filename=lung_path.name),
                *[
                    dict(type="csv_file", filename=nodule_file.name)
                    for nodule_file in self._nodule_files
                ],
            ],
            "error_messages": [],
        }

    def predict(
        self, lung_image: SimpleITK.Image, nodules_locations: DataFrame
    ) -> DataFrame:
        scores = []
        lung_data = SimpleITK.GetArrayFromImage(lung_image)
        for idx, nodule in nodules_locations.iterrows():
            coord = lung_image.TransformPhysicalPointToIndex(
                (nodule["coordX"], nodule["coordY"], nodule["coordZ"])
            )
            val = lung_data[coord[2], coord[1], coord[0]]
            scores.append(val)
        nodules_locations.loc[:, "class"] = scores
        return nodules_locations


def test_detection_evaluation(tmpdir):
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

    expected_results_file = (
        Path(__file__).parent / "resources" / "json" / "results.json"
    )

    with open(expected_results_file, "r") as f:
        expected_result = json.load(f)

    assert results == expected_result
