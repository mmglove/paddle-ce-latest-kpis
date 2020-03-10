#!/bin/bash
# This file is only used for continuous evaluation.

export CUDA_VISIBLE_DEVICES=1
if [ -d 'models' ]; then
    rm -rf models
fi
python local_train.py --enable_ce --model_output_dir models --num_epoch 1 >xdeepfm_T_gpu1.log 2>&1
cat xdeepfm_T_gpu1.log|grep data: |awk -F "[" 'END{print "kpis\ttest_auc_gpu1\t"$2}' |tr -d ']' | python _ce.py

export CUDA_VISIBLE_DEVICES=1,2
if [ -d 'models' ]; then
    rm -rf models
fi
python local_train.py --enable_ce --model_output_dir models --num_epoch 1 >xdeepfm_T_gpu2.log 2>&1
cat xdeepfm_T_gpu2.log|grep data: |awk -F "[" 'END{print "kpis\ttest_auc_gpu2\t"$2}' |tr -d ']' | python _ce.py
#  infer
model=xdeepfm_infer
CUDA_VISIBLE_DEVICES=1 python infer.py --model_output_dir models --test_epoch 1 --use_gpu 1 >xdeepfm_infer_gpu.log 2>&1
if [ $? -ne 0 ];then
	echo -e "${model},FAIL"
else
	echo -e "${model},SUCCESS"
fi