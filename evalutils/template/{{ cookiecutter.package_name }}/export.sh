#!/usr/bin/env bash

./build.sh

docker save {{ cookiecutter.package_name|lower }} > {{ cookiecutter.package_name }}.tar
