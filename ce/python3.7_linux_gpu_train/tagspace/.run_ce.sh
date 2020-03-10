#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

cudaid=${tagspace:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train.py --enable_ce --use_cuda 1 --train_dir train_big_data/ --vocab_text_path big_vocab_text.txt --vocab_tag_path big_vocab_tag.txt --model_dir big_model --batch_size 500 --parallel 1 1> log_1card
cat log_1card | python _ce.py

cudaid=${tagspace_4:=0,1,2,3} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train.py --enable_ce --use_cuda 1 --train_dir train_big_data/ --vocab_text_path big_vocab_text.txt --vocab_tag_path big_vocab_tag.txt --model_dir big_model --batch_size 500 --parallel 1 1> log_4cards
cat log_4cards | python _ce.py
# infer gpu
model=tag_I_gpu
CUDA_VISIBLE_DEVICES=0 python infer.py --use_cuda 1 --model_dir big_model --vocab_tag_path big_vocab_tag.txt --test_dir test_big_data --start_index 1 --last_index 1 >tagspace_infer_all.log 2>&1
if [ $? -ne 0 ];then
	echo -e "${model},FAIL"
else
	echo -e "${model},SUCCESS"
fi
