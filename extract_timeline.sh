#!/bin/bash
video_file=$1
set -f
set -x
video_prefix=${video_file//.mp4/}
array=(${video_prefix//\// })
len=${#array[@]}

outpath=$2/${array[-1]}
mkdir -p $outpath

./build/extract_gpu -f=$video_file -x=$outpath/flow_x -y=$outpath/flow_y -i=$outpath/flow_i -b=20 -t=0 -d=0 -s=1 -o=dir -w=0 -h=0
