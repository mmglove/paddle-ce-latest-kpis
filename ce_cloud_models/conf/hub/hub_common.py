# encoding: utf-8
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
本文件提供了代码中用到的公共配置。
Creators: paddlepaddle-qa
Date:    2021/02/17 14:33:27
"""

import os


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
# 公共全局变量
PADDLE_ON_MODEL_CE = "1"
WITH_AVX = "ON"
# 资源配置
IS_SINGLE_CUDA = True
XPU = 'gpu'  # 取值gpu或cpu

SET_CUDA = '0'
SET_MULTI_CUDA = '0,1'

# PaddleHub
REPO_PaddleHub = "https://github.com/PaddlePaddle/PaddleHub.git"
# BASE_BRANCH = "release/v2.1"
BASE_BRANCH = "develop"

NLP_BRANCH = BASE_BRANCH

EXEC_PRIORITY = ["p0", "p1"]
EXEC_CASES = ["SETUP_HUB", "INSTALL", "TEXTMATCHING_FINETUNE", "TEXTMATCHING_PREDICT"]
EXEC_TAG = [
    "linux_st_gpu1",
    "linux_dy_gpu1",
    "linux_st_gpu1_con",
    "linux_dy_gpu1_con",
    # 补充一些自定义标签
    "linux_setup_hub",
    "linux_install_module",
    "linux_finetune_gpu1",
    "linux_finetune_all_gpu1"
]
