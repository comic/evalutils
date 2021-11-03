#!/usr/bin/env bash
set -o errexit

SCRIPTPATH="$(dirname "$(realpath "${0}")")"

docker build -t {{ cookiecutter.package_name|lower }} "${SCRIPTPATH}"
