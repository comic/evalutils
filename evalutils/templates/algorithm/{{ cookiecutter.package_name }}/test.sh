#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
{% if cookiecutter.template_kind == "Algorithm" -%}
# Maximum is currently 30g, configurable in your algorithm image settings on grand challenge
{%- endif %}
MEM_LIMIT="4g"

docker volume create {{ cookiecutter.package_name|lower }}-output-$VOLUME_SUFFIX

# Do not change any of the parameters to docker run, these are fixed
docker run --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test/:/input/ \
        -v {{ cookiecutter.package_name|lower }}-output-$VOLUME_SUFFIX:/output/ \
        {{ cookiecutter.package_name|lower }}

docker run --rm \
        -v {{ cookiecutter.package_name|lower }}-output-$VOLUME_SUFFIX:/output/ \
        python:{{ cookiecutter.python_major_version }}.{{ cookiecutter.python_minor_version }}-slim cat /output/results.json | python -m json.tool

{% if cookiecutter.template_kind == "Algorithm" -%}
docker run --rm \
        -v {{ cookiecutter.package_name|lower }}-output-$VOLUME_SUFFIX:/output/ \
        -v $SCRIPTPATH/test/:/input/ \
        python:{{ cookiecutter.python_major_version }}.{{ cookiecutter.python_minor_version }}-slim python -c "import json, sys; f1 = json.load(open('/output/results.json')); f2 = json.load(open('/input/expected_output.json')); sys.exit(f1 != f2);"

if [ $? -eq 0 ]; then
    echo "Tests successfully passed..."
else
    echo "Expected output was not found..."
fi
{%- endif %}

docker volume rm {{ cookiecutter.package_name|lower }}-output-$VOLUME_SUFFIX
