#!/usr/bin/env bash

./build.sh

docker save {{ cookiecutter.package_name|lower }} | gz -c > {{ cookiecutter.package_name }}.tar.gz
