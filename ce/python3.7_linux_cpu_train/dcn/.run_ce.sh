#!/bin/bash
# This file is only used for continuous evaluation.
rm -rf *_factor.txt
export CPU_NUM=1
if [ -d 'models' ]; then
    rm -rf models
fi
python local_train.py --enable_ce  --num_thread 1 --steps 1000 >dcn_cpu1_thread1.log 2>&1
cat dcn_cpu1_thread1.log|grep data: |awk -F "[" 'END{print "kpis\ttest_auc_cpu1_thread1\t"$2}' |tr -d ']' | python _ce.py

export CPU_NUM=1
if [ -d 'models' ]; then
    rm -rf models
fi
python local_train.py --enable_ce  --num_thread 20 --steps 1000 >dcn_cpu1_thread20.log 2>&1
cat dcn_cpu1_thread20.log|grep data: |awk -F "[" 'END{print "kpis\ttest_auc_cpu1_thread20\t"$2}' |tr -d ']' | python _ce.py

model=dcn_infer
python infer.py --test_epoch 1 >dcn_infer_e1.log 2>&1
if [ $? -ne 0 ];then
	echo -e "${model},FAIL"
else
	echo -e "${model},SUCCESS"
fi