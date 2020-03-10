#!/bin/bash
# This file is only used for continuous evaluation.

export CPU_NUM=1
if [ -d 'models' ]; then
    rm -rf models
fi
python local_train.py --enable_ce --num_thread 10 --model_output_dir models --num_epoch 1 --use_gpu 0 >xdeepfm_cpu1_thread1.log 2>&1
cat xdeepfm_cpu1_thread1.log|grep data: |awk -F "[" 'END{print "kpis\ttest_auc_cpu1_thread1\t"$2}' |tr -d ']' | python _ce.py

export CPU_NUM=1
if [ -d 'models' ]; then
    rm -rf models
fi
python local_train.py --enable_ce --num_thread 10 --model_output_dir models --num_epoch 1 --use_gpu 0 >xdeepfm_cpu1_thread10.log 2>&1
cat xdeepfm_cpu1_thread10.log|grep data: |awk -F "[" 'END{print "kpis\ttest_auc_cpu1_thread10\t"$2}' |tr -d ']' | python _ce.py
#  infer
model=xdeepfm_infer
python infer.py --model_output_dir models --test_epoch 1 --use_gpu 0 >xdeepfm_infer_cpu.log 2>&1
if [ $? -ne 0 ];then
	echo -e "${model},FAIL"
else
	echo -e "${model},SUCCESS"
fi
