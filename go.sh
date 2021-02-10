#!/usr/bin/env bash

die() { printf $'Error: %s\n' "$*" >&2; exit 1; }

root=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
build=${root:?}/build
venv=${root:?}/venv
spack=${root:?}/spack
data=${root:?}/data
checkpoint=${root:?}/checkpoint
basetag=horovod/horovod:0.20.0-tf2.3.0-torch1.6.0-mxnet1.5.0-py3.7-cpu
tag=horovod_$USER
registry=accona.eecs.utk.edu:5000
python=$(which python3.8)

[ -f ${root:?}/env.sh ] && . ${root:?}/env.sh

go-spack() {
    if ! [ -x ${spack:?}/bin/spack ]; then
        git clone https://github.com/spack/spack.git ${spack:?} >&2 || die "Could not clone spack"
    fi

    exec ${spack:?}/bin/spack "$@"
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
    if ! [ -x ${venv:?}/bin/python ]; then
        ${python:?} -m virtualenv -p ${python:?} ${venv:?} || die "Cannot setup virtualenv"
    fi
    ${venv:?}/bin/"$@"
}

go-cmake() {
    : ${SPACK_ENV:?I need to be run in a Spack environment}
    cmake -H"${root:?}" -B"${build:?}" \
        -DCMAKE_CXX_COMPILER=mpicxx \
        -DCMAKE_C_COMPILER=gcc \
        "$@"
}

go-make() {
    : ${SPACK_ENV:?I need to be run in a Spack environment}
    make -C "${build:?}" \
        VERBOSE=1 \
        "$@"
}

go-exec() {
    : ${SPACK_ENV:?I need to be run in a Spack environment}
    exec "$@"
}

go-env() {
    eval $(go-spack env activate --sh --dir ${root:?})
    exec env "$@"
}

go-clean() {
    if [ $# -eq 0 ]; then
        set -- data spack venv
    fi
    for arg; do
        case "$arg" in
        (data) rm -rf ${data:?};;
        (spack) rm -rf ${spack:?} ${root:?}/.spack-env;;
        (venv) rm -rf {venv:?};;
        esac
    done
}

go-redirect() (
    local opt OPTIND OPTARG opt_stdout opt_stderr opt_stdin
    opt_stdout=
    opt_stderr=
    opt_stdin=
    while getopts "o:e:i:" opt; do
        case "$opt" in
        (o) opt_stdout=$OPTARG;;
        (e) opt_stderr=$OPTARG;;
        (i) opt_stdin=$OPTARG;;
        esac
    done
    shift $((OPTIND-1))

    if [ -n "$opt_stdout" ]; then
        exec 1>"$opt_stdout"
    fi

    if [ -n "$opt_stderr" ]; then
        exec 2>"$opt_stderr"
    fi

    if [ -n "$opt_stdin" ]; then
        exec 0<"$opt_stdin"
    fi

    exec "$@"
)

go-trial() {
    : ${SPACK_ENV:?I need to be run in a Spack environment}
    ! [ "${python#${SPACK_ENV:?}}" = "${python:?}" ] || die "Expected ${python} to start with ${SPACK_ENV}"

    exec > >(tee -a stdout.txt)
    exec 2> >(tee -a stderr.txt >/dev/stderr)

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

    columns=( dataset div num_conv_layers nepochs1 nworkers1 mode )
    printf $'%s,' "${columns[@]}"
    printf $'real,user,sys\n'

    for dataset in emnist; do
    for div in 100; do
    for num_conv_layers in 2; do
    for nepochs1 in 10; do
    for nworkers1 in 5; do
    for nepochs2 in ${nepochs1}; do
    for nworkers2 in ${nworkers1}; do
    for events in "${nepochs1}e/nworkers=${nworkers1} ${nepochs2}e/nworkers=${nworkers2},reload=True"; do

    for mode in test; do

    for c in "${columns[@]}"; do
        printf $'%s,' "${!c}" | tee /dev/stderr
    done

    rm -rf "${checkpoint:?}"
    mkdir "${checkpoint:?}"
    
    /usr/bin/time \
    --format='%e,%U,%S' \
        ${root:?}/go.sh redirect \
        -e /dev/stdout \
            $(which mpirun) \
            -np 5 \
            -hostfile ${root:?}/hostlist.txt \
            -iface eno1 \
                    ${venv:?}/bin/python \
                    -u \
                        triple-r.py \
                        --data-dir ${data:?} \
                        --checkpoint-dir ${checkpoint:?} \
                        --default-verbosity 2 \
                        --num-conv-layers ${num_conv_layers} \
                        --dataset ${dataset} \
                        --div ${div} \
                        ${events} \
    3>&1 4>&2 2>&3- 1>&4- | tee /dev/stderr

    done

    done
    done
    done
    done
    done
    done
    done
    done
}

go-docker() {
    local arg args
    args=()
    for arg; do
        arg=${arg//\$tag/$tag}
        arg=${arg//\$basetag/$basetag}
        arg=${arg//\$registry/$registry}
        args+=( "$arg" )
    done
    exec docker "${args[@]}"
}

go-process-log() {
    _extract_from_end() {
        local index_from_end
        index_from_end=${1:?}
        awk -v x=${index_from_end} '
            /^==== Date: / { ++n }
            { L[n, ++M[n]] = $0 }
            END {
                for(i=1; i<=M[n-x]; ++i)
                    print L[n-x, i]
            }
        '
    }

    _extract_csv() {
        awk '
            BEGIN {
                FS=OFS=",";
                n = 0;
            }
            /^emnist,/ {
                split($0, ary, FS);
                H[n++] = ary[1] OFS ary[2] OFS ary[3] OFS ary[4] OFS ary[5];
                M[n] = 0;
            }
            match($0, /^Epoch ([0-9]+)\/([0-9]+)/, ary) {
                E[n, M[n]] = ary[1];
            }
            match($0, /^stats = loss=([0-9.]+) accuracy=([0-9.]+)/, ary) {
                L[n, M[n]] = ary[1];
                A[n, M[n]] = ary[2];
                M[n]++;
            }
            END {
                print "dataset", "num_conv_layers", "nepochs1", "nworkers1", "mode", "epoch", "loss", "accuracy";
                for (i=0; i<n; ++i)
                    for (j=0; j<M[i]; ++j)
                        print H[i], E[i, j], L[i, j], A[i, j];
            }
        '
    }

    _extract_from_end ${1:?need index from end} | _extract_csv
}

go-"$@"
