#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File generate_TFRecord.py
# @ Description :
# @ Author alexchung
# @ Time 2020/06/01 PM 15:48

import os
import numpy as np
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from tensorflow.python_io import tf_record_iterator
from tensorflow.python.ops import control_flow_ops
from DataProcess.inception_preprocessing import preprocess_image

dataset_dir = '/home/alex/Documents/dataset/flower_tfrecord'

train_data_path = os.path.join(dataset_dir, 'train')
test_data_path = os.path.join(dataset_dir, 'val')


def parse_example(serialized_sample, target_shape, class_depth, is_training=False):
    """
    parse tensor
    :param image_sample:
    :return:
    """

    # construct feature description
    image_feature_description ={

        "image": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "label": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "height": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "width": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "depth": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "filename": tf.io.FixedLenFeature(shape=[], dtype=tf.string)
    }
    feature = tf.io.parse_single_example(serialized=serialized_sample, features=image_feature_description)

    # parse feature
    raw_img = tf.decode_raw(feature['image'], tf.uint8)
    # shape = tf.cast(feature['shape'], tf.int32)
    height = tf.cast(feature['height'], tf.int32)
    width = tf.cast(feature['width'], tf.int32)
    depth = tf.cast(feature['depth'], tf.int32)

    image = tf.reshape(raw_img, [height, width, depth])
    label = tf.cast(feature['label'], tf.int32)
    filename = tf.cast(feature['filename'], tf.string)
    # resize image shape
    # random crop image
    # before use shuffle_batch, use random_crop to make image shape to special size
    # first step enlarge image size
    # second step dataset operation

    # image augmentation
    # image = augmentation_image(image=image, image_shape=input_shape, preprocessing_type=preprocessing_type,
    #                            fast_mode=fast_mode, is_training=is_training,)
    image = preprocess_image(image=image, height=target_shape[0], width=target_shape[1], is_training=is_training)
    # onehot label
    label = tf.one_hot(indices=label, depth=class_depth)

    return image, label, filename


def augmentation_image(image, image_shape, flip_lr=False, flip_ud=False, brightness=False,
                       bright_delta=32. / 255., contrast=False, contrast_lower=0.5, contrast_up=1.5, hue=False,
                       hue_delta=0.2, saturation=False, saturation_low=0.5, saturation_up=1.5, fast_mode=True,
                       preprocessing_type='vgg', is_training = False,):
    """

    :param image:
    :param image_shape:
    :param flip_lr:
    :param flip_ud:
    :param brightness:
    :param bright_delta:
    :param contrast:
    :param contrast_lower:
    :param contrast_up:
    :param hue:
    :param hue_delta:
    :param saturation:
    :param saturation_low:
    :param saturation_up:
    :param fast_mode:
    :param preprocessing_type: vgg | inception | cifar | lenet
    :param is_training:
    :return:
    """
    try:

        if preprocessing_type == "vgg":
            # resize image
            # resize_img = aspect_preserve_resize(input_image, resize_side_min=np.rint(image_shape[0] * 1.04),
            #                                    resize_side_max=np.rint(image_shape[0] * 2.08), is_training=is_training)
            resize_img = aspect_preserve_resize(image, resize_side_min=256,
                                                resize_side_max=288, is_training=is_training)

            # crop image
            distort_img = image_crop(resize_img, image_shape[0], image_shape[1], is_training = is_training)

            if is_training:
                # enlarge image to same size

                # flip image in left and right
                if flip_lr:
                    distort_img = tf.image.random_flip_left_right(image=distort_img, seed=0)
                # flip image in left and right
                if flip_ud:
                    distort_img = tf.image.random_flip_up_down(image=distort_img, seed=0)

            return distort_img

        elif preprocessing_type == "inception":
            if image.dtype != tf.float32:
                # convert image scale  to [0, 1]
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)

            if is_training:
                # random crop image

                # create bbox for crop image
                bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
                # Each bounding box has shape [1, num_boxes, box coords] and
                # the coordinates are ordered [ymin, xmin, ymax, xmax].
                distort_img, distort_bbox = distorted_bounding_box_crop(image, bbox)

                # reshape image
                distort_img.set_shape([None, None, 3])

                # resize image with random method
                num_resize_cases = 1 if fast_mode else 4
                distort_img = apply_with_random_selector(distort_img,
                    lambda x, method: tf.image.resize_images(x, [image_shape[0], image_shape[0]], method),
                    num_cases=num_resize_cases)

                if fast_mode:
                    # Randomly flip the image horizontally.
                    distort_img = tf.image.random_flip_left_right(distort_img, seed=0)
                else:
                    if flip_lr:
                        distort_img = tf.image.random_flip_left_right(image=distort_img, seed=0)
                    # flip image in left and right
                    if flip_ud:
                        distort_img = tf.image.random_flip_up_down(image=distort_img, seed=0)
                    # adjust image brightness
                    if brightness:
                        distort_img = tf.image.random_brightness(image=distort_img, max_delta=bright_delta)
                    # # adjust image contrast
                    if contrast:
                        distort_img = tf.image.random_contrast(image=distort_img, lower=contrast_lower, upper=contrast_up)
                    # adjust image hue
                    if hue:
                        distort_img = tf.image.random_hue(image=distort_img, max_delta=hue_delta)
                    #  adjust image saturation
                    if saturation:
                        distort_img = tf.image.random_saturation(image=distort_img, lower=saturation_low,
                                                                upper=saturation_up)
            # eval preprocess
            else:
                image = tf.image.central_crop(image, central_fraction=0.875)

                image = tf.expand_dims(image, 0)
                image = tf.image.resize_bilinear(image, [image_shape[0], image_shape[1]], align_corners=False)
                distort_img = tf.squeeze(image, [0])

            return distort_img

    except Exception as e:
        print('\nFailed augmentation for {0}'.format(e))



#--------------------------------------for VGG preprocessing-----------------------------------------------
def aspect_preserve_resize(image, resize_side_min=256, resize_side_max=512, is_training=False):
    """

    :param image_tensor:
    :param output_height:
    :param output_width:
    :param resize_side_min:
    :param resize_side_max:
    :return:
    """
    if is_training:
        smaller_side = tf.random_uniform([], minval=resize_side_min, maxval=resize_side_max, dtype=tf.float32)
    else:
        smaller_side = resize_side_min

    shape = tf.shape(image)

    height, width = tf.cast(shape[0], dtype=tf.float32), tf.cast(shape[1], dtype=tf.float32)

    resize_scale = tf.cond(pred=tf.greater(height, width),
                           true_fn=lambda : smaller_side / width,
                           false_fn=lambda : smaller_side / height)

    new_height = tf.cast(tf.rint(height * resize_scale), dtype=tf.int32)
    new_width = tf.cast(tf.rint(width * resize_scale), dtype=tf.int32)

    resize_image = tf.image.resize(image, size=(new_height, new_width))

    return tf.cast(resize_image, dtype=image.dtype)


def image_crop(image, output_height=224, output_width=224, is_training=False):
    """

    :param image:
    :param output_height:
    :param output_width:
    :param is_training:
    :return:
    """
    shape = tf.shape(image)
    depth = shape[2]
    if is_training:

        crop_image = tf.image.random_crop(image, size=(output_height, output_width, depth))
    else:
        crop_image = central_crop(image, output_height, output_width)

    return tf.cast(crop_image, image.dtype)

def central_crop(image, crop_height=224, crop_width=224):
    """
    image central crop
    :param image:
    :param output_height:
    :param output_width:
    :return:
    """

    shape = tf.shape(image)
    height, width, depth = shape[0], shape[1], shape[2]


    offset_height = (height - crop_height) / 2
    offset_width = (width - crop_width) / 2

    # assert image rank must be 3
    rank_assertion = tf.Assert(tf.equal(tf.rank(image), 3), ['Rank of image must be equal 3'])

    with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, depth])

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(height, crop_height),
            tf.greater_equal(width, crop_width)),
        ['Image size greater than the crop size'])

    offsets = tf.cast(tf.stack([offset_height, offset_width, 0]), dtype=tf.int32)

    with tf.control_dependencies([size_assertion]):
        # crop with slice
        crop_image = tf.slice(image, begin=offsets, size=cropped_shape)

    return tf.reshape(crop_image, cropped_shape)


#-------------------------------------for inception prepprocessing--------------------------------------------------

def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].
          Args:
            x: input Tensor.
            func: Python function to apply.
            num_cases: Python int32, number of cases to sample sel from.
          Returns:
            The result of func(x, sel), where func receives the value of the
            selector as a python integer, but sel is sampled dynamically.
          """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
        for case in range(num_cases)])[0]


def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
    """Generates cropped_image using a one of the bboxes randomly distorted.
       See `tf.image.sample_distorted_bounding_box` for more documentation.
       Args:image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
          bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
          where each coordinate is [0, 1) and the coordinates are arranged
          as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
          image.
        min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
          area of the image must contain at least this fraction of any bounding box
          supplied.
        aspect_ratio_range: An optional list of `floats`. The cropped area of the
          image must have an aspect ratio = width / height within this range.
        area_range: An optional list of `floats`. The cropped area of the image
          must contain a fraction of the supplied image within in this range.
        max_attempts: An optional `int`. Number of attempts at generating a cropped
          region of the image of the specified constraints. After `max_attempts`
          failures, return the entire image.
        scope: Optional scope for name_scope.
      Returns:
        A tuple, a 3-D Tensor cropped_image and the distorted bbox
      """
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].

        # A large fraction of image datasets contain a human-annotated bounding
        # box delineating the region of the image containing the object of interest.
        # We choose to create a new bounding box for the object which is a randomly
        # distorted version of the human-annotated bounding box that obeys an
        # allowed range of aspect ratios, sizes and overlap with the human-annotated
        # bounding box. If no box is supplied, then we assume the bounding box is
        # the entire image.
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        return cropped_image, distort_bbox



def dataset_tfrecord(record_files, target_shape, class_depth, epoch=5, batch_size=10, shuffle=True, is_training=False):
    """
    construct iterator to read image
    :param record_file:
    :return:
    """
    # check record file format
    if os.path.isfile(record_files):
        record_list = [record_files]
    else:
        record_list = [os.path.join(record_files, record_file) for record_file in os.listdir(record_files)
                       if record_file.split('.')[-1] == 'record']
    # # use dataset read record file
    raw_img_dataset = tf.data.TFRecordDataset(record_list)
    # execute parse function to get dataset
    # This transformation applies map_func to each element of this dataset,
    # and returns a new dataset containing the transformed elements, in the
    # same order as they appeared in the input.
    # when parse_example has only one parameter (office recommend)
    # parse_img_dataset = raw_img_dataset.map(parse_example)
    # when parse_example has more than one parameter which used to process data
    parse_img_dataset = raw_img_dataset.map(lambda series_record:
                                            parse_example(series_record, target_shape, class_depth,
                                                          is_training=is_training))
    # get dataset batch
    if shuffle:
        shuffle_batch_dataset = parse_img_dataset.shuffle(buffer_size=batch_size*4).repeat(epoch).batch(batch_size=batch_size)
    else:
        shuffle_batch_dataset = parse_img_dataset.repeat(epoch).batch(batch_size=batch_size)
    # make dataset iterator
    image, label, filename = shuffle_batch_dataset.make_one_shot_iterator().get_next()

    # image = augmentation_image(input_image=image, image_shape=input_shape)
    # # onehot label
    # label = tf.one_hot(indices=label, depth=class_depth)

    return image, label, filename


def reader_tfrecord(record_files, target_shape, class_depth, batch_size=10, num_threads=2, epoch=5, shuffle=True,
                    is_training=False):
    """
    read and sparse TFRecord
    :param record_file:
    :return:
    """
    record_list = []
    # check record file format
    if os.path.isfile(record_files):
        record_list = [record_files]
    else:
        record_list = [os.path.join(record_files, record_file) for record_file in os.listdir(record_files)
                       if record_file.split('.')[-1] == 'record']
    # create input queue
    filename_queue = tf.train.string_input_producer(string_tensor=record_list, num_epochs=epoch, shuffle=shuffle)
    # create reader to read TFRecord sample instant
    reader = tf.TFRecordReader()
    # read one sample instant
    _, serialized_sample = reader.read(filename_queue)

    # parse sample
    image, label, filename = parse_example(serialized_sample, target_shape=target_shape, class_depth=class_depth,
                                           is_training=is_training)

    if shuffle:
        image, label, filename = tf.train.shuffle_batch([image, label, filename],
                                          batch_size=batch_size,
                                          capacity=batch_size * 4,
                                          num_threads=num_threads,
                                          min_after_dequeue=batch_size)
    else:
        image, label, filename = tf.train.batch([image, label, filename],
                                                batch_size=batch_size,
                                                capacity=batch_size,
                                                num_threads=num_threads,
                                                enqueue_many=False
                                                )
    # dataset = tf.data.Dataset.shuffle(buffer_size=batch_size*4)
    return image, label, filename


def get_num_samples(record_dir):
    """
    get tfrecord numbers
    :param record_file:
    :return:
    """

    record_list = [os.path.join(record_dir, record_file) for record_file in os.listdir(record_dir)
                   if record_file.split('.')[-1] == 'record']

    num_samples = 0
    for record_file in record_list:
        for _ in tf_record_iterator(record_file):
            num_samples += 1
    return num_samples

if __name__ == "__main__":
    num_samples = get_num_samples(train_data_path)
    print('all sample size is {0}'.format(num_samples))
    image_batch, label_batch, filename = dataset_tfrecord(record_files=train_data_path, target_shape=[224, 224, 3],
                                                          class_depth=5, is_training=True)

    # create local and global variables initializer group
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        sess.run(init_op)

        # create Coordinator to manage the life period of multiple thread
        coord = tf.train.Coordinator()
        # Starts all queue runners collected in the graph to execute input queue operation
        # the step contain two operation:filename to filename queue and sample to sample queue
        threads = tf.train.start_queue_runners(coord=coord)
        print('threads: {0}'.format(threads))
        try:
            if not coord.should_stop():
                image_feed, label_feed = sess.run([image_batch, label_batch])
                plt.imshow(image_feed[0])
                plt.show()
        except Exception as e:
            print(e)
        finally:
            # request to stop all background threads
            coord.request_stop()

        # waiting all threads safely exit
        coord.join(threads)
        sess.close()

