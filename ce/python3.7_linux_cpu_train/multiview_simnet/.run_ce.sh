#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1


export CPU_NUM=1
export NUM_THREADS=1

FLAGS_benchmark=true  python train.py --enable_ce 1> log
cat log | python _ce.py

#infer
model='multiview_simnet_infer'
python infer.py >multiview_I.log 2>&1
if [ $? -ne 0 ];then
	echo -e "${model},FAIL"
else
	echo -e "${model},SUCCESS"
fi