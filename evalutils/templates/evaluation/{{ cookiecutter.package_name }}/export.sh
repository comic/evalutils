#!/usr/bin/env bash

./build.sh

docker save {{ cookiecutter.package_name|lower }} | gzip -c > {{ cookiecutter.package_name }}.tar.gz
