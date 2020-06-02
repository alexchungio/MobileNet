#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : mobilenet_v2.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/6/1 下午3:27
# @ Software   : PyCharm
#-------------------------------------------------------


import tensorflow as tf
import tensorflow.contrib.slim as slim
import copy

from MobileNet_V2_Tensorflow.libs import conv_blocks as ops
import MobileNet_V2_Tensorflow.libs.mobilenet as lib

op = lib.op
expand_input = ops.expand_input_by_factor


# pyformat: disable
# Architecture: https://arxiv.org/abs/1801.04381
V2_DEF = dict(
    defaults={
        # Note: these parameters of batch norm affect the architecture
        # that's why they are here and not in training_scope.
        (slim.batch_norm,): {'center': True, 'scale': True},
        (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
            'normalizer_fn': slim.batch_norm, 'activation_fn': tf.nn.relu6
        },
        (ops.expanded_conv,): {
            'expansion_size': expand_input(6),
            'split_expansion': 1,
            'normalizer_fn': slim.batch_norm,
            'residual': True
        },
        (slim.conv2d, slim.separable_conv2d): {'padding': 'SAME'}
    },
    spec=[
        op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
        op(ops.expanded_conv,
           expansion_size=expand_input(1, divisible_by=1),
           num_outputs=16),
        op(ops.expanded_conv, stride=2, num_outputs=24),
        op(ops.expanded_conv, stride=1, num_outputs=24),
        op(ops.expanded_conv, stride=2, num_outputs=32),
        op(ops.expanded_conv, stride=1, num_outputs=32),
        op(ops.expanded_conv, stride=1, num_outputs=32),
        op(ops.expanded_conv, stride=2, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=96),
        op(ops.expanded_conv, stride=1, num_outputs=96),
        op(ops.expanded_conv, stride=1, num_outputs=96),
        op(ops.expanded_conv, stride=2, num_outputs=160),
        op(ops.expanded_conv, stride=1, num_outputs=160),
        op(ops.expanded_conv, stride=1, num_outputs=160),
        op(ops.expanded_conv, stride=1, num_outputs=320),
        op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280)
    ],
)
# pyformat: enable


class MobileNetV2():
    """
    VGG16 model
    """
    def __init__(self, input_shape, num_classes, batch_size, decay_rate, learning_rate, depth_multiplier=1.0,
                 keep_prob=0.8, weight_decay=0.00004, num_samples_per_epoch=None, num_epoch_per_decay=None,
                 is_training=False):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.decay_steps = int(num_samples_per_epoch * num_epoch_per_decay / batch_size)
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.depth_multiplier = depth_multiplier
        # self.optimizer = optimizer
        self.keep_prob = keep_prob
        self.weight_decay = weight_decay
        self.is_training = is_training

        # compatible last layer size of small multipliers
        if self.depth_multiplier < 1.0:
            self.fine_grain = True
        else:
            self.fine_grain = False

        self.raw_input_data = tf.placeholder(tf.float32, shape=[None, input_shape[0], input_shape[1], input_shape[2]],
                                             name="input_images")
        # self.raw_input_data = self.mean_subtraction(image=self.raw_input_data,
        #                                             means=[self._R_MEAN, self._G_MEAN, self._B_MEAN])
        self.raw_input_label = tf.placeholder(tf.float32, shape=[None, self.num_classes], name="class_label")

        self.global_step = tf.train.get_or_create_global_step()
        # self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        # self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")

        # logits
        self.logits =  self.inference(inputs=self.raw_input_data, name='MobilenetV2')
        # computer loss value
        self.loss = self.losses(labels=self.raw_input_label, logits=self.logits, name='loss')
        # train operation
        self.train = self.training(self.learning_rate, self.global_step)
        self.accuracy = self.get_accuracy(logits=self.logits, labels=self.raw_input_label)

    def inference(self, inputs, name):
        """
        vgg16 inference
        construct static map
        :param input_op:
        :return:
        """

        with tf.variable_scope(name, reuse=None) as sc:
            prop = self.mobilenet_v2(inputs=inputs,
                                   num_classes= self.num_classes,
                                   is_training = self.is_training,
                                   depth_multiplier=self.depth_multiplier,
                                   finegrain_classification_mode = self.fine_grain,
                                   scope=sc)

        return prop

    @slim.add_arg_scope
    def mobilenet_v2(self, inputs,
                     num_classes=1001,
                     depth_multiplier=1.0,
                     scope='MobilenetV2',
                     conv_defs=None,
                     finegrain_classification_mode=False,
                     min_depth=None,
                     divisible_by=None,
                     activation_fn=None,
                     **kwargs):
        """Creates mobilenet V2 network.

        Inference mode is created by default. To create training use training_scope
        below.

        with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
           logits, endpoints = mobilenet_v2.mobilenet(input_tensor)

        Args:
          input_tensor: The input tensor
          num_classes: number of classes
          depth_multiplier: The multiplier applied to scale number of
          channels in each layer.
          scope: Scope of the operator
          conv_defs: Allows to override default conv def.
          finegrain_classification_mode: When set to True, the model
          will keep the last layer large even for small multipliers. Following
          https://arxiv.org/abs/1801.04381
          suggests that it improves performance for ImageNet-type of problems.
            *Note* ignored if final_endpoint makes the builder exit earlier.
          min_depth: If provided, will ensure that all layers will have that
          many channels after application of depth multiplier.
          divisible_by: If provided will ensure that all layers # channels
          will be divisible by this number.
          activation_fn: Activation function to use, defaults to tf.nn.relu6 if not
            specified.
          **kwargs: passed directly to mobilenet.mobilenet:
            prediction_fn- what prediction function to use.
            reuse-: whether to reuse variables (if reuse set to true, scope
            must be given).
        Returns:
          logits/endpoints pair

        Raises:
          ValueError: On invalid arguments
        """
        with tf.contrib.slim.arg_scope(lib.training_scope(weight_decay=self.weight_decay,
                                                          dropout_keep_prob=self.keep_prob,
                                                          is_training=False)):
            if conv_defs is None:
                conv_defs = V2_DEF
            if 'multiplier' in kwargs:
                raise ValueError('mobilenetv2 doesn\'t support generic '
                                 'multiplier parameter use "depth_multiplier" instead.')
            if finegrain_classification_mode:
                conv_defs = copy.deepcopy(conv_defs)
                if depth_multiplier < 1:
                    conv_defs['spec'][-1].params['num_outputs'] /= depth_multiplier
            if activation_fn:
                conv_defs = copy.deepcopy(conv_defs)
                defaults = conv_defs['defaults']
                conv_defaults = (
                    defaults[(slim.conv2d, slim.fully_connected, slim.separable_conv2d)])
                conv_defaults['activation_fn'] = activation_fn

            depth_args = {}
            # NB: do not set depth_args unless they are provided to avoid overriding
            # whatever default depth_multiplier might have thanks to arg_scope.
            if min_depth is not None:
                depth_args['min_depth'] = min_depth
            if divisible_by is not None:
                depth_args['divisible_by'] = divisible_by

            with slim.arg_scope((lib.depth_multiplier,), **depth_args):
                logits, end_points = lib.mobilenet(inputs,
                                                   num_classes=num_classes,
                                                   conv_defs=conv_defs,
                                                   scope=scope,
                                                   multiplier=depth_multiplier,
                                                   **kwargs)
                prob = end_points['Predictions']
                return prob


    def training(self, learning_rate, global_step, trainable_scope=None):
        """
        train operation
        :param learnRate:
        :param globalStep:
        :param args:
        :return:
        """
        # define trainable variable
        # define frozen layer

        if trainable_scope is not None:
            trainable_variable = []
            for scope in trainable_scope:
                variables = tf.model_variables(scope=scope)
                [trainable_variable.append(var) for var in variables]
        else:
            trainable_variable = None

        learning_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step,
                                                   decay_steps=self.decay_steps, decay_rate=self.decay_rate,
                                                   staircase=False)
        # # according to use request of slim.batch_norm
        # # update moving_mean and moving_variance when training
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss, global_step=global_step,
                                                                                 var_list=trainable_variable)
            return train_op

    def load_weights(self, sess, model_path, custom_scope=None):
        """
        load pre train model
        :param sess:
        :param model_path:
        :param custom_scope:
        :return:
        """

        model_variable = tf.model_variables()
        if custom_scope is None:
            custom_scope = ['MobilenetV2/Logits/Conv2d_1c_1x1']
        for scope in custom_scope:
            variables = tf.model_variables(scope=scope)
            [model_variable.remove(var) for var in variables]
        saver = tf.train.Saver(var_list=model_variable)
        saver.restore(sess, save_path=model_path)
        print('Successful load pretrain model from {0}'.format(model_path))

    # def predict(self):
    #     """
    #     predict operation
    #     :return:
    #     """
    #
    #     return tf.cast(self.logits, dtype=tf.float32, name="predicts")

    def losses(self, logits, labels, name):
        """
        loss function
        :param logits:
        :param labels:
        :return:
        """
        with tf.name_scope(name):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='entropy')
            loss = tf.reduce_mean(input_tensor=cross_entropy, name='loss')
            # tf.losses.add_loss(loss) # add normal loss to losses collection
            # # weight_loss = slim.losses.get_regularization_losses()
            # # tf.losses.add_loss(weight_loss) # add regularization loss to losses collection
            # total_loss = tf.losses.get_total_loss()
            weight_loss = tf.add_n(slim.losses.get_regularization_losses())
            total_loss = loss + weight_loss
            tf.summary.scalar("total loss", total_loss)
            return total_loss

    def get_accuracy(self, logits, labels):
        """
        evaluate one batch correct num
        :param logits:
        :param label:
        :return:
        """
        correct_predict = tf.equal(tf.argmax(input=logits, axis=1), tf.argmax(input=labels, axis=1))
        return tf.reduce_mean(tf.cast(correct_predict, dtype=tf.float32))

    def fill_feed_dict(self, image_feed, label_feed):
        feed_dict = {
            self.raw_input_data: image_feed,
            self.raw_input_label: label_feed,
        }
        return feed_dict