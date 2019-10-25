#!/bin/bash
file_name=$1
root_dir=$2
set -f
set -x
video_prefix=${root_dir}
array=(${video_prefix//\// })
len=${#array[@]}

outpath=$3/${array[-1]}
mkdir -p $outpath

./build/extract_frames_gpu -f=$file_name -r=$root_dir -or=$outpath -b=20 -t=0 -d=0 -o=dir -w=0 -h=0