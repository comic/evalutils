#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

VOLUME_SUFFIX=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)

docker volume create {{ cookiecutter.package_name|lower }}-output-$VOLUME_SUFFIX

docker run --rm \
        --memory=4g \
        -v $SCRIPTPATH/test/:/input/ \
        -v {{ cookiecutter.package_name|lower }}-output-$VOLUME_SUFFIX:/output/ \
        {{ cookiecutter.package_name|lower }}

docker run --rm \
        -v {{ cookiecutter.package_name|lower }}-output-$VOLUME_SUFFIX:/output/ \
        {{ cookiecutter.docker_base_container }} cat /output/metrics.json | python -m json.tool

docker volume rm {{ cookiecutter.package_name|lower }}-output-$VOLUME_SUFFIX
