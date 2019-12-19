#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t {{ cookiecutter.package_name|lower }} "$SCRIPTPATH"
