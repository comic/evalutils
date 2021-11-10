#!/usr/bin/env bash

# set -o errexit

# RANDOM_READ_BYTES is tha amount of bytes to read from /dev/urandom.
# No particular reason for any specific value, just something vaguely sensible that provides a
# finite limit on reading from /dev/urandom and gives the message digest (here hard-coded to md5)
# something to calculate on...
RANDOM_READ_BYTES=1024
SCRIPTPATH="$(dirname "$(realpath "${0}")")"

# echo "*** run ${SCRIPTPATH}/build.sh" >> "${TMPLOG}"
# "${SCRIPTPATH}/build.sh"

PACKAGE_NAME={{ cookiecutter.package_name|lower }}
VOLUME_SUFFIX=$(dd if=/dev/urandom bs=${RANDOM_READ_BYTES} count=1 | md5sum | cut --delimiter=' ' --fields=1)
VOLUME_NAME="${PACKAGE_NAME}-output-${VOLUME_SUFFIX}"

echo "*** vol name ${VOLUME_NAME} ; docker vol create: "
docker volume create "${VOLUME_NAME}"

echo "*** last exec: $? : docker run"
docker run --rm \
        --memory=4g \
        -v "${SCRIPTPATH/test/}":/input/ \
        -v "${VOLUME_NAME}":/output/ \
        "${PACKAGE_NAME}"   # cmd missing here? Or am I just mis-reading this?
echo "*** last exec: $? : docker run cat json output"
docker run --rm \
        -v "${VOLUME_NAME}":/output/ \
        "${PACKAGE_NAME}" cat /output/metrics.json | python -m json.tool

echo "*** last exec: $? : docker volume rm"
docker volume rm "${VOLUME_NAME}"
echo "*** last exec: $? : done"
