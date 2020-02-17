#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

docker volume create {{ cookiecutter.package_name|lower }}-output

docker run --rm \
        --memory={{ cookiecutter.requirements.memory|lower }} \
        -v $SCRIPTPATH/test/:/input/ \
        -v {{ cookiecutter.package_name|lower }}-output:/output/ \
        {{ cookiecutter.package_name|lower }}

docker run --rm \
        -v {{ cookiecutter.package_name|lower }}-output:/output/ \
        {{ cookiecutter.docker_base_container }} cat /output/results.json | python -m json.tool

docker run --rm \
        -v {{ cookiecutter.package_name|lower }}-output:/output/ \
        -v $SCRIPTPATH/test/:/input/ \
        {{ cookiecutter.docker_base_container }} python -c "import json, sys; f1 = json.load(open('/output/results.json')); f2 = json.load(open('/input/expected_output.json')); sys.exit(f1 != f2);"

if [ $? -eq 0 ]; then
    echo "Tests successfully passed..."
else
    echo "Expected output was not found..."
fi

docker volume rm {{ cookiecutter.package_name|lower }}-output
