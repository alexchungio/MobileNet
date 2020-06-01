#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : MobileNet_v2_demo.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/6/1 上午10:58
# @ Software   : PyCharm
#------------------------------------------------------

import os
import tensorflow as tf
from enum import Enum


class DepthMultiplier(Enum):
    """
    depth multiplier
    """
    multiplier_0 = 0.5
    multiplier_1 = 0.75
    multiplier_2 = 1.0
    multiplier_3 = 1.5

pertrain_model = '/home/alex/Documents/pretrain_model/mobilenet_v2'
model_path = os.path.join(pertrain_model, f'mobilenet_v2_{DepthMultiplier.multiplier_0.value}_224')
meta_graph = os.path.join(model_path, f'mobilenet_v2_{DepthMultiplier.multiplier_0.value}_224.ckpt.meta')


if __name__ == "__main__":

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        pass


