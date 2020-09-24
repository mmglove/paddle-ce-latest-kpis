#!/bin/bash
current_dir=`PWD`
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
print_info(){
if [ $1 -ne 0 ];then
    mv ${log_path}/$2 ${log_path}/$2_FAIL.log
    echo -e "\033[31m ${log_path}/$2_FAIL \033[0m"
else
    mv ${log_path}/$2 ${log_path}/$2_SUCCESS.log
    echo -e "\033[32m ${log_path}/$2_SUCCESS \033[0m"
fi
}
# contentunderstanding
# CPU,自备全量的yaml文件拷贝到当前目录下，全量数据待拷贝

mkdir logs && mkdir
export log_path=logs
run_con_cpu(){
cp ${dataset_path}/rec_configs/$1_cpu_config.yaml ./
python -m paddlerec.run -m ./$1_cpu_config.yaml >${log_path}/$1_cpu 2>&1
#print_info $? $1
}
# 1.1 tagspace
model=tagspace
cd ${current_dir}/models/contentunderstanding/tagspace
ln -s ${dataset_path}/textcnn/senta_data senta_data
run_con_cpu ${model}
cat ${log_path}/${model}_cpu |grep done |head -10|tail -1 |awk -F ' |,|=' '{print ""'${model}'"_epoch_time_cpu\t"$7"\t"'${model}'"_acc_cpu\t"$12"\t"'${model}'"_loss_cpu\t"$15}'|tr -d '[][]' |python _ce.py
# 1.2 textcnn
model=textcnn
cd ${current_dir}/models/contentunderstanding/textcnn
ln -s ${dataset_path}/textcnn/senta_data senta_data
run_con_cpu ${model}
cat ${log_path}/${model}_cpu |grep done |head -16|tail -1 |awk -F ' |,|=' '{print ""'${model}'"_epoch_time_cpu\t"$7"\t"'${model}'"_acc_cpu\t"$12"\t"'${model}'"_loss_cpu\t"$15}'|tr -d '[][]' |python _ce.py
# 1.3 textcnn_pretrain
model=textcnn_pretrain
cd ${current_dir}/models/contentunderstanding/textcnn
ln -s ${dataset_path}/textcnn/senta_data senta_data
run_con_cpu ${model}
cat ${log_path}/${model}_cpu |grep done |head -6|tail -1 |awk -F ' |,|=' '{print ""'${model}'"_epoch_time_cpu\t"$7"\t"'${model}'"_acc_cpu\t"$12"\t"'${model}'"_loss_cpu\t"$15}'|tr -d '[][]' |python _ce.py

# 3.1 multitask (1/3)  esmm
model=esmm
run_con_cpu ${model}
cat ${log_path}/${model}_cpu |grep done |head -6|tail -1 |awk -F ' |,|=' '{print ""'${model}'"_epoch_time_cpu\t"$7"\t"'${model}'"_acc_cpu\t"$12"\t"'${model}'"_loss_cpu\t"$15}'|tr -d '[][]' |python _ce.py


cudaid1=${card1:=2} # use 0-th card as default
# textcnn的下的全量数据需要自备all_yaml,需要确认是否加CPU和GPU的不同的，目前都是默认CPU的
run_con_gpu(){
CUDA_VISIBLE_DEVICES=$cudaid1 python -m paddlerec.run -m ./rec_config/$1_cpu_config.yaml >${log_path}/$1_gpu1 2>&1
#print_info $? $1
}
content_models=' textcnn textcnn_pretrain'
# 1.1 tagspace
model=tagspace
cp ${data_path}/con_configs/${model}_cpu_config.yaml ./models/contentunderstanding/${model}/
ln -s ${data_path}/textcnn/
run_con_cpu ${model}
cat ${log_path}/${model}_cpu |grep done |head -10|tail -1 |awk -F ' |,|=' '{print ""'${model}'"_epoch_time_cpu\t"$7"\t"'${model}'"_acc_cpu\t"$12"\t"'${model}'"_loss_cpu\t"$15}'|tr -d '[][]' |python _ce.py
