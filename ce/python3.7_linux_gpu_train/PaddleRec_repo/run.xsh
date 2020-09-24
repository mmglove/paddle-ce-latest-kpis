#!/bin/bash
export rec_dir=$PWD/../../rec_repo
#copy models files
cp -r ${rec_dir}/PaddleRec/. ./
# 需要处理所有的data数据，或者在yaml 文件中修改好,目前还没有
cp ${dataset_path}/rec_config ./
./.run_ce.sh
