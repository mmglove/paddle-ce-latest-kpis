#!/bin/bash
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

export CPU_NUM=1
export NUM_THREADS=1

FLAGS_benchmark=true  python train.py --train_dir train_big_data --vocab_path vocab_big.txt --use_cuda 0 --batch_size 500 --model_dir model_output --pass_num 2 --enable_ce --step_num 10 >log_cpu
cat log_cpu | python _ce.py

model='gru4rec_infer'
python infer.py --test_dir test_big_data/ --model_dir model_output --start_index 1 --last_index 1 --use_cuda 0 >gru4rec_infer_cpu.log 2>&1
if [ $? -ne 0 ];then
	echo -e "${model},FAIL"
else
	echo -e "${model},SUCCESS"
fi
