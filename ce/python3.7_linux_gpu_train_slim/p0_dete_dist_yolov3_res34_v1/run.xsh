#!/bin/bash
export detection_dir=$PWD/../../detection_repo
export slim_dir=$PWD/../../slim_repo
cd ${slim_dir}
pip install -r requirements.txt
python setup.py install
cd -
#copy PaddleDetection files
cp -r ${detection_dir}/. ./

if [ -d "dataset" ];then rm -rf dataset
fi
ln -s ${dataset_path}/yolov3/dataset dataset

./.run_ce.sh
