#!/usr/bin/env bash

die() { printf $'Error: %s\n' "$*" >&2; exit 1; }

root=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
build=${root:?}/build
venv=${root:?}/venv
spack=${root:?}/spack
python=$(which python3.8)

[ -f ${root:?}/env.sh ] && . ${root:?}/env.sh

go-spack() {
    if ! [ -d ${spack:?} ]; then
        git clone https://github.com/spack/spack.git ${spack:?} >&2 || die "Could not clone spack"
    fi

    exec ./spack/bin/spack "$@"
}

go-venv() {
    : ${SPACK_ENV:?I need to be run in a Spack environment}
    ! [ "${python#${SPACK_ENV:?}}" = "${python:?}" ] || die "Expected ${python} to start with ${SPACK_ENV}"
    if ! ${python:?} -c 'import virtualenv' &>/dev/null; then
        if ! ${python:?} -c 'import pip' &>/dev/null; then
            if ! ${python:?} -c 'import ensurepip' &>/dev/null; then
                die "Cannot import ensurepip"
            fi
            ${python:?} -m ensurepip || die "Cannot ensurepip"
        fi
        ${python:?} -m pip install --user virtualenv || die "Cannot install virtualenv"
    fi
    if ! [ -d ${venv:?} ]; then
        ${python:?} -m virtualenv -p ${python:?} ${venv:?} || die "Cannot setup virtualenv"
    fi
    ${venv:?}/bin/pip install -r requirements.txt || die "Cannot pip install requirements.txt"
}

go-cmake() {
    : ${SPACK_ENV:?I need to be run in a Spack environment}
    : ${VIRTUAL_ENV:?I need to be run in a Python virtualenv}
    cmake -H"${root:?}" -B"${build:?}" \
        -DCMAKE_CXX_COMPILER=mpicxx \
        -DCMAKE_C_COMPILER=gcc \
        "$@"
}

go-make() {
    : ${SPACK_ENV:?I need to be run in a Spack environment}
    : ${VIRTUAL_ENV:?I need to be run in a Python virtualenv}
    make -C "${build:?}" \
        VERBOSE=1 \
        "$@"
}

go-exec() {
    : ${SPACK_ENV:?I need to be run in a Spack environment}
    : ${VIRTUAL_ENV:?I need to be run in a Python virtualenv}
    exec "$@"
}

go-trial() {
    : ${SPACK_ENV:?I need to be run in a Spack environment}
    : ${VIRTUAL_ENV:?I need to be run in a Python virtualenv}
    ! [ "${python#${SPACK_ENV:?}}" = "${python:?}" ] || die "Expected ${python} to start with ${SPACK_ENV}"

    exec > >(tee -a results.txt)
    exec 2> >(tee -a log.txt)

    printf $'==== Date: %s\n' "$(date)" >&2
    printf $'==== File: %s\n' triple-r.py >&2
    cat triple-r.py >&2
    printf $'==== File: go.sh\n' >&2
    cat go.sh >&2
    printf $'==== File: requirements.txt\n' >&2
    cat requirements.txt >&2
    printf $'==== File: spack.yaml\n' >&2
    cat spack.yaml >&2
    printf $'==== Args:' >&2
    printf $' %q' "$@" >&2
    printf $'\n' >&2
    printf $'====\n' >&2
    
    /usr/bin/time --format='%e,%U,%S' \
        ${python:?} triple-r.py \
            "$@" \
        2>&1 | tee /dev/stderr
}

go-"$@"
