#!/usr/bin/env bash

./build.sh

TEMPDIR=`mktemp -d`

docker run -v $(pwd)/test/:/input/ \
           -v $TEMPDIR:/output/ \
           {{ cookiecutter.package_name|lower }}

cat $TEMPDIR/metrics.json

rm -rf $TEMPDIR
