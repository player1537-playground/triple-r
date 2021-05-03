#!/usr/bin/env bash

die() { printf $'Error: %s\n' "$*" >&2; exit 1; }

root=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
build=${root:?}/build
venv=${root:?}/venv
spack=${root:?}/spack
data=${root:?}/data
logs=${root:?}/logs
jobs=${root:?}/jobs
checkpoint=${root:?}/checkpoint
horovod=${root:?}/horovod
basetag=horovod/horovod:0.20.0-tf2.3.0-torch1.6.0-mxnet1.5.0-py3.7-cpu
tag=horovod_$USER
registry=accona.eecs.utk.edu:5000
whatreallyhappened=${root:?}/_whatreallyhappened
host=
python=$(which python3.8 2>/dev/null)
simg=${root:?}/tensorflow-20.12-tf2-py3.simg
stag=docker://nvcr.io/nvidia/tensorflow:20.12-tf2-py3
scache=${root:?}/scache
stmp=${root:?}/stmp

[ -f ${root:?}/env.sh ] && . ${root:?}/env.sh

go-singularity() {
    go-singularity-"$@"
}

go-singularity-build() {
    SINGULARITY_CACHEDIR=${scache:?} \
    SINGULARITY_TMPDIR=${stmp:?} \
    singularity build \
        ${simg:?} \
        ${stag:?}
}

go-singularity-exec() {
    singularity exec \
        --nv \
        -B /soft,/gpfs/mira-home/thobson,/home,/lus,/scratch \
        ${simg:?} \
        ${root:?}/go.sh \
        "$@"
}

go-spack() {
    if ! [ -x ${spack:?}/bin/spack ]; then
        git clone https://github.com/spack/spack.git ${spack:?} >&2 || die "Could not clone spack"
    fi

    exec ${spack:?}/bin/spack "$@"
}

go-horovod() {
    : ${SPACK_ENV:?I need to be run in a Spack environment}
    : ${VIRTUAL_ENV:?need to be run inside a virtual env}
    if ! [ -f ${horovod:?}/setup.py ]; then
        git clone --recursive https://github.com/horovod/horovod.git ${horovod:?}
    fi

    if [ $# -gt 0 ]; then
        (cd ${horovod:?} && "$@")
    fi
}

go-whatreallyhappened() {
    : ${SPACK_ENV:?I need to be run in a Spack environment}
    #: ${VIRTUAL_ENV:?need to be run inside a virtual env}
    if ! [ -f ${whatreallyhappened:?}/go.sh ]; then
        git clone https://github.com/player1537-playground/whatreallyhappened.git ${whatreallyhappened:?}
    fi

    if [ $# -gt 0 ]; then
        (. ${venv:?}/bin/activate && cd ${whatreallyhappened:?} && ./go.sh "$@")
    fi
}

go-venv() {
    : ${SPACK_ENV:?I need to be run in a Spack environment}
    #! [ "${python#${SPACK_ENV:?}}" = "${python:?}" ] || die "Expected ${python} to start with ${SPACK_ENV}"
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
    #: ${SPACK_ENV:?I need to be run in a Spack environment}
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

go-cobalt() {
    LD_PRELOAD= \
    time \
        ${root:?}/go.sh \
            trial
}

go-genjobs() {
    go-onetrial() {
        njobs=$(ls -lah ${jobs:?}/*.sh 2>/dev/null | wc -l)
        cat <<EOF >jobs/job_${njobs}.sh
#!/usr/bin/env bash
#COBALT -t $((2 * 60))
#COBALT -n ${nworkers:?}
#COBALT -q default
#COBALT --attrs=pubnet:nox11
#COBALT -O ${jobs:?}/job_${njobs}

dataset=${dataset:?}
model=${model:?}
div=${div:?}
nepochs=${nepochs:?}
nworkers=${nworkers:?}
ckpt_freq=${ckpt_freq:?}
failure_epoch=${failure_epoch:?}
mode=${mode:?}
name=${name:?}

. ${root:?}/go.sh onetrial
EOF
        chmod +x ${jobs:?}/job_${njobs}.sh
    }

    mkdir -p ${jobs:?}
    go-trial
}

go-onetrial() {
    events=()
    did_failure=0
    since_last_ckpt=0
    for ((i=1; i<=nepochs; ++i)); do
        event="1e/nworkers=${nworkers:?},seed=${seed:?}"
        if ((!did_failure && i == failure_epoch)); then
            if [ ${#events[@]} -gt 0 ]; then
                events[${#events[@]}-1]+=",checkpoint=True"
            fi
            event+=",reload=True"
            did_failure=1
            (( i -= since_last_ckpt ))
        fi
        (( since_last_ckpt++ ))
        if ((ckpt_freq != 0 && i % ckpt_freq == 0)); then
            event+=",checkpoint=True"
            since_last_ckpt=0
        fi
        events+=( "${event}" )
    done
    OIFS=$IFS
    IFS=$' '
    events="${events[*]}"
    IFS=$OIFS

    rm -rf "${checkpoint:?}"
    mkdir "${checkpoint:?}"

    sleep 1 || return 1

    mkdir -p ${logs:?}/${name:?}

    mkdir -p /dev/shm/metem/data
    (cd /dev/shm/metem/data && tar xf ${data:?}/tiny-imagenet-200.tar.gz)
    
    $(which mpirun) \
    -np ${nworkers} \
    -host ${host:?} \
    ${iface:+-iface ${iface:?}} \
        ${root:?}/go.sh singularity exec env \
        HOROVOD_TIMELINE=${root:?}/timeline.json \
            ${whatreallyhappened:?}/go.sh exec \
                ${venv:?}/bin/python \
                -u \
                    triple-r.py \
                    --dataset ${dataset:?} \
                    --model ${model:?} \
                    --data-dir /dev/shm/metem/data \
                    --checkpoint-dir ${checkpoint:?} \
                    --default-verbosity 2 \
                    --div ${div} \
                    --log-to /dev/stdout \
                    ${events}
    #>&2
                    #--log-to ${logs:?}/${name:?}/'%(rank+1)dof%(size)d.log' \
}

go-trial() {
    #: ${SPACK_ENV:?I need to be run in a Spack environment}
    #! [ "${python#${SPACK_ENV:?}}" = "${python:?}" ] || die "Expected ${python} to start with ${SPACK_ENV}"

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

    columns=( dataset model div nepochs nworkers ckpt_freq failure_epoch mode )
    printf $'%s,' "${columns[@]}"
    printf $'real,user,sys\n'

    for nepochs in 8; do
    for ckpt_freq in 4 3 2 1 0; do
    for failure_epoch in 2 5; do
    for dataset in tiny-imagenet; do
    for div in 1; do
    for model in ResNet50; do
    for nworkers in 8; do
    for mode in test; do
    for name in immediate_backward_recovery,ckpt_freq=${ckpt_freq:?},failure_epoch=${failure_epoch:?}; do

    for c in "${columns[@]}"; do
        printf $'%s,' "${!c}" | tee /dev/stderr
    done

    go-onetrial || return 1

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

go-stress-checkpoint() {
    mkdir -p /dev/shm/metem/checkpoint
    mkdir -p /scratch/metem/checkpoint

    printf $'checkpoint_dir,iterations,model,format,log_to,mode,bytes,real,user,sys\n' | tee -a checkpoint.csv >&2

    for checkpoint_dir in /scratch/metem/checkpoint; do
    for model in Long-32-{10..50..10} Wide-{10240..51200..10240}; do
    for iterations in 100; do
    for format in hdf5 tensorflow; do
    for log_to in /lus/theta-fs0/projects/VeloC/metem/logs/log.log; do

    case "${format:?}" in
    (hdf5)
        iterations=100
        ;;
    (tensorflow)
        iterations=25
        ;;
    esac

    for mode in prime test test test; do

    printf $'%s,%s,%s,%s,%s,%s,' ${checkpoint_dir:?} ${iterations:?} ${model:?} ${format:?} ${log_to:?} ${mode:?} | tee -a checkpoint.csv >&2

    sleep 0.5 || break

    LD_PRELOAD= \
    /usr/bin/time \
    --format='%e,%U,%S' \
        ${root:?}/go.sh redirect \
        -e /dev/stdout \
            ${root:?}/go.sh singularity exec \
                env \
                    ${whatreallyhappened:?}/go.sh \
                        exec \
                            ${venv:?}/bin/python ${root:?}/checkpoint.py \
                            --iterations ${iterations:?} \
                            --checkpoint-dir ${checkpoint_dir:?} \
                            --log-to ${log_to:?} \
                            --model ${model:?} \
                            --format ${format:?} \
    2> >(tee -a checkpoint.csv >&2) \
            100>&2

    done

    done
    done
    done
    done
    done
}

go-extract-tiny-imagenet() {
	# Train looks like:
	#   data/tiny-imagenet-200/train/
	#   data/tiny-imagenet-200/train/n02437312
	#   data/tiny-imagenet-200/train/n02437312/images
	#   data/tiny-imagenet-200/train/n02437312/images/n02437312_273.JPEG
	#   data/tiny-imagenet-200/train/n02437312/images/n02437312_192.JPEG
	#   data/tiny-imagenet-200/train/n02437312/images/n02437312_418.JPEG
	#   data/tiny-imagenet-200/train/n02437312/images/n02437312_404.JPEG
	#   data/tiny-imagenet-200/train/n02437312/images/n02437312_30.JPEG
	#   data/tiny-imagenet-200/train/n02437312/images/n02437312_269.JPEG
	#   data/tiny-imagenet-200/train/n02437312/images/n02437312_239.JPEG

	# Validation looks like:
	#   data/tiny-imagenet-200/val/
	#   data/tiny-imagenet-200/val/val_annotations.txt
	#   data/tiny-imagenet-200/val/images
	#   data/tiny-imagenet-200/val/images/val_9447.JPEG
	#   data/tiny-imagenet-200/val/images/val_8152.JPEG
	#   data/tiny-imagenet-200/val/images/val_9676.JPEG
	#   data/tiny-imagenet-200/val/images/val_2518.JPEG
	#   data/tiny-imagenet-200/val/images/val_251.JPEG
	#   data/tiny-imagenet-200/val/images/val_6638.JPEG
	#   data/tiny-imagenet-200/val/images/val_8046.JPEG

	# val_annotations.txt looks like:
	#   val_0.JPEG      n03444034       0       32      44      62
	#   val_1.JPEG      n04067472       52      55      57      59
	#   val_2.JPEG      n04070727       4       0       60      55
	#   val_3.JPEG      n02808440       3       3       63      63
	#   val_4.JPEG      n02808440       9       27      63      48
	#   val_5.JPEG      n04399382       7       0       59      63
	#   val_6.JPEG      n04179913       0       0       63      56
	#   val_7.JPEG      n02823428       5       0       57      63
	#   val_8.JPEG      n04146614       0       31      60      60
	#   val_9.JPEG      n02226429       0       3       63      57

	if ! [ -f ${data:?}/tiny-imagenet-200/val.bak/val_annotations.txt ]; then
		mv \
			${data:?}/tiny-imagenet-200/val \
			${data:?}/tiny-imagenet-200/val.bak
	fi

	exec < ${data:?}/tiny-imagenet-200/val.bak/val_annotations.txt
	while IFS=$'\t' read -r filename class bbox0 bbox1 bbox2 bbox3; do
		orig=${data:?}/tiny-imagenet-200/val.bak/images/${filename:?}
		new=${data:?}/tiny-imagenet-200/val/${class:?}/images/${filename:?}

		mkdir -p ${new%/*}
		ln -sf ${orig:?} ${new:?}
	done
}

go-"$@"
