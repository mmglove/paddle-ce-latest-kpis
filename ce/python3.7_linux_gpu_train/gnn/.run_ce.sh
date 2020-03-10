#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1


cudaid=${gnn:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python -u train.py --use_cuda 1 --epoch_num 5 --enable_ce 1> log
cat log | python _ce.py

model='gnn_infer'
CUDA_VISIBLE_DEVICES=0 python infer.py --last_index 1 --use_cuda 1 >gnn_infer.log 2>&1
if [ $? -ne 0 ];then
	echo -e "${model},FAIL"
else
	echo -e "${model},SUCCESS"
fi
