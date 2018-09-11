#!/usr/bin/env bash

./build.sh

docker volume create {{ cookiecutter.package_name|lower }}-output

docker run --rm \
        -v $(pwd)/test/:/input/ \
        -v {{ cookiecutter.package_name|lower }}-output:/output/ \
        {{ cookiecutter.package_name|lower }}

docker run --rm \
        -v {{ cookiecutter.package_name|lower }}-output:/output/ \
        alpine:latest cat /output/metrics.json

docker volume rm {{ cookiecutter.package_name|lower }}-output
