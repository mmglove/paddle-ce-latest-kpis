#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

export CPU_NUM=1
export NUM_THREADS=1

FLAGS_benchmark=true  python train.py --train_dir train_big_data --vocab_path vocab_big.txt --use_cuda 0 --batch_size 500 --model_dir model_output --epochs 1 --enable_ce --step_num 500 1> log_cpu
cat log_cpu | python _ce.py

model='ssr_infer'
python infer.py --test_dir test_big_data --vocab_path vocab_big.txt --use_cuda 0 --batch_size 50 --model_dir model_output --last_index 1 >ssr_infer.log 2>&1
if [ $? -ne 0 ];then
	echo -e "${model},FAIL"
else
	echo -e "${model},SUCCESS"
fi
