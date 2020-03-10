#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

export NUM_THREADS=1

rm -rf v1_cpu5_b100_lr1dir
export OPENBLAS_NUM_THREADS=1
export CPU_NUM=1
FLAGS_benchmark=true python train.py --enable_ce --train_data_dir data/convert_text8 --dict_path data/test_build_dict --num_passes 1 --batch_size 100 --model_output_dir v1_cpu1_demo_model --base_lr 1.0 --print_batch 1000 --with_speed --is_sparse >word2vec_trian_demo_cpu1.log 2>&1
cat word2vec_trian_demo_cpu1.log|grep 880000 |awk -F ' ' 'NR==1{print "kpis\ttest_cpu1_loss\t"$10}' | python _ce.py

rm -rf v1_cpu5_b100_lr1dir
export OPENBLAS_NUM_THREADS=1
export CPU_NUM=5
FLAGS_benchmark=true python train.py --enable_ce --train_data_dir data/convert_text8 --dict_path data/test_build_dict --num_passes 1 --batch_size 100 --model_output_dir v1_cpu1_demo_model --base_lr 1.0 --print_batch 1000 --with_speed --is_sparse >word2vec_trian_demo_cpu5.log 2>&1
cat word2vec_trian_demo_cpu5.log|grep 176000 |awk -F ' ' 'NR==1{print "kpis\ttest_cpu5_loss\t"$10}' | python _ce.py

#  infer
model=word2vec_infer
python infer.py --infer_epoch --test_dir data/test_mid_dir --dict_path data/test_build_dict_word_to_id_ --batch_size 20000 --model_dir v1_cpu5_b100_lr1dir/  --start_index 0 --last_index 0 >word2vec_I_demo_gpu.log 2>&1
if [ $? -ne 0 ];then
	echo -e "${model},FAIL"
else
	echo -e "${model},SUCCESS"
fi
