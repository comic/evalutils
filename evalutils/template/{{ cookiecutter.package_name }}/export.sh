#!/usr/bin/env bash

./build.sh

docker save {{ cookiecutter.package_name }} > {{ cookiecutter.package_name }}.tar
