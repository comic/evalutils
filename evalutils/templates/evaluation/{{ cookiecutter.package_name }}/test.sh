#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)

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
