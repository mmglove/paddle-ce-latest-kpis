#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES=0
export CPU_NUM=1
export NUM_THREADS=1

FLAGS_benchmark=true  python -u train.py --config_path 'data/config.txt' --train_dir 'data/paddle_train.txt' --batch_size 32 --epoch_num 1 --use_cuda 0 --enable_ce --batch_num 10000 >log_cpu
cat log_cpu |  python _ce.py

model=din_infer
python infer.py --model_path 'din_amazon/global_step_50000' --test_path 'data/paddle_test.txt' --use_cuda 0 >din_infer_cpu.log 2>&1
if [ $? -ne 0 ];then
	echo -e "${model},FAIL"
else
	echo -e "${model},SUCCESS"
fi