# Copyright UCL Business plc 2017. Patent Pending. All rights reserved. 
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence, 
# please contact info@uclb.com

"""Stereo data loader.
"""

from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
from PIL import Image
from stereo_utils import *

def string_length_tf(t):
    return tf.py_func(len, [t], [tf.int64])

def load_pfm_wrap(p):
    data, _ = load_pfm(p, False)
    #print('data: {} {} {}'.format(data.dtype, data.shape, np.mean(data)))
    return data

def decode_webp_wrap(p):
    # return np.array np.uint8 (H, W, C)
    img = Image.open(p)
    img = np.asarray(img)
    #print('img: {} {}'.format(img.dtype, img.shape))
    return img

def load_pfm_tf(t):
    return tf.py_func(load_pfm_wrap, [t], [tf.float32])

def decode_webp_tf(t):
    return tf.py_func(decode_webp_wrap, [t], [tf.uint8])


class StereoDataloader(object):
    """stereo dataloader"""

    def __init__(self, data_path, filenames_file, params, dataset, mode, supervised=True):
        self.data_path = data_path
        self.params = params
        self.dataset = dataset
        self.mode = mode

        self.left_image_batch  = None
        self.right_image_batch = None
        self.disp_image_batch = None

        has_disp = (supervised and not mode == 'inference')

        input_queue = tf.train.string_input_producer([filenames_file], shuffle=False)
        line_reader = tf.TextLineReader()
        _, line = line_reader.read(input_queue)

        split_line = tf.string_split([line]).values

        if mode == 'inference':
            left_image_path  = tf.string_join([self.data_path, split_line[0]])
            right_image_path = tf.string_join([self.data_path, split_line[1]])
            left_image_o  = self.read_image(left_image_path)
            right_image_o = self.read_image(right_image_path)

        elif mode == 'train' or mode == 'test':
            left_image_path  = tf.string_join([self.data_path, split_line[0]])
            right_image_path = tf.string_join([self.data_path, split_line[1]])
            if has_disp:
                disp_image_path = tf.string_join([self.data_path, split_line[2]])

            left_image_o  = self.read_image(left_image_path)
            right_image_o = self.read_image(right_image_path)
            if has_disp:
                disp_image_o = self.read_disparity(disp_image_path)

        else:
            print('Unknown mode: {}'.format(mode))
            exit()


        if params.data_augment and mode == 'train':
            # randomly flip images
            do_flip = tf.random_uniform([], 0, 1)
            left_image  = tf.cond(do_flip > 0.5, lambda: tf.image.flip_up_down(left_image_o), lambda: left_image_o)
            right_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_up_down(right_image_o),  lambda: right_image_o)
            if has_disp:
                disp_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_up_down(disp_image_o),  lambda: disp_image_o)

            # randomly augment images
            do_augment  = tf.random_uniform([], 0, 1)
            left_image, right_image = tf.cond(do_augment > 0.5, lambda: self.augment_image_pair(left_image, right_image), lambda: (left_image, right_image))

        else:
            left_image = left_image_o
            right_image = right_image_o
            if has_disp:
                disp_image = disp_image_o

        if mode == 'train':
            left_image.set_shape( [None, None, 3])
            right_image.set_shape([None, None, 3])
            if has_disp:
                disp_image.set_shape([None, None, 1])

            # capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
            min_after_dequeue = 256
            capacity = min_after_dequeue + 4 * params.batch_size
            if has_disp:
                self.left_image_batch, self.right_image_batch, self.disp_image_batch = tf.train.shuffle_batch([left_image, right_image, disp_image],
                        batch_size = params.batch_size, capacity=capacity, min_after_dequeue = min_after_dequeue, num_threads=params.num_threads)
            else:
                self.left_image_batch, self.right_image_batch = tf.train.shuffle_batch([left_image, right_image],
                        batch_size = params.batch_size, capacity=capacity, min_after_dequeue = min_after_dequeue, num_threads=params.num_threads)


        elif mode == 'test':
            left_image_o.set_shape( [None, None, 3])
            right_image_o.set_shape([None, None, 3])
            if has_disp:
                disp_image_o.set_shape([None, None, 1])

            self.left_image_batch = tf.stack([left_image_o], 0)
            self.right_image_batch = tf.stack([right_image_o], 0)
            if has_disp:
                self.disp_image_batch = tf.stack([disp_image_o], 0)
        elif mode == 'inference':
            self.left_image_batch = tf.stack([left_image_o], 0)
            self.right_image_batch = tf.stack([right_image_o], 0)

        # do random crop
        if has_disp:
            merged = tf.concat([self.left_image_batch, self.right_image_batch, self.disp_image_batch], -1)
            croped = tf.random_crop(merged, (tf.shape(merged)[0], params.crop_height, params.crop_width, 7))
            self.left_image_batch, self.right_image_batch, self.disp_image_batch = croped[:,:,:,0:3], croped[:,:,:,3:6], croped[:,:,:,6:7]
        
            self.left_image_batch = tf.reshape(self.left_image_batch, [self.params.batch_size, self.params.crop_height, self.params.crop_width, 3])
            self.right_image_batch = tf.reshape(self.right_image_batch, [self.params.batch_size, self.params.crop_height, self.params.crop_width, 3])
            self.disp_image_batch = tf.reshape(self.disp_image_batch, [self.params.batch_size, self.params.crop_height, self.params.crop_width, 1])
        else:
            merged = tf.concat([self.left_image_batch, self.right_image_batch], -1)
            croped = tf.random_crop(merged, (tf.shape(merged)[0], params.crop_height, params.crop_width, 6))
            self.left_image_batch, self.right_image_batch = croped[:,:,:,0:3], croped[:,:,:,3:6]
        
            self.left_image_batch = tf.reshape(self.left_image_batch, [self.params.batch_size, self.params.crop_height, self.params.crop_width, 3])
            self.right_image_batch = tf.reshape(self.right_image_batch, [self.params.batch_size, self.params.crop_height, self.params.crop_width, 3])

    def augment_image_pair(self, left_image, right_image):
        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        left_image_aug  = left_image  ** random_gamma
        right_image_aug = right_image ** random_gamma

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        left_image_aug  =  left_image_aug * random_brightness
        right_image_aug = right_image_aug * random_brightness

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        white = tf.ones([tf.shape(left_image)[0], tf.shape(left_image)[1]])
        color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
        left_image_aug  *= color_image
        right_image_aug *= color_image

        # saturate
        left_image_aug  = tf.clip_by_value(left_image_aug,  0, 1)
        right_image_aug = tf.clip_by_value(right_image_aug, 0, 1)

        return left_image_aug, right_image_aug

    def read_image(self, image_path):
        path_length = string_length_tf(image_path)[0]
        file_extension = tf.substr(image_path, path_length - 4, 4)
        file_cond = tf.equal(file_extension, 'webp')

        image = tf.cond(file_cond, lambda: decode_webp_tf(image_path)[0], lambda: tf.image.decode_image(tf.read_file(image_path), channels=3))
        #image.set_shape([self.height, self.width, 3])
        image = tf.image.convert_image_dtype(image,  tf.float32)
        image = tf.image.resize_image_with_crop_or_pad(image, self.params.height, self.params.width)

        return image
    
    def read_disparity(self, image_path):
        path_length = string_length_tf(image_path)[0]
        file_extension = tf.substr(image_path, path_length - 3, 3)
        file_cond = tf.equal(file_extension, 'png')

        #disp = tf.cond(file_cond,   lambda: tf.image.convert_image_dtype(tf.image.decode_png(tf.read_file(image_path), dtype=tf.uint16), tf.float32)/256.0, 
        disp = tf.cond(file_cond,   lambda: tf.cast(tf.image.decode_png(tf.read_file(image_path), channels=1, dtype=tf.uint16), tf.float32)/256.0, 
                                    lambda: tf.expand_dims(load_pfm_tf(image_path)[0], -1))
        #disp.set_shape([self.params.height, self.params.width, 1])
        disp = tf.image.resize_image_with_crop_or_pad(disp, self.params.height, self.params.width)

        return disp



if __name__ == '__main__':
    from collections import namedtuple
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    #import matplotlib.pyplot as plt
    stereo_parameters = namedtuple('parameters', 
                        'encoder, '
                        'height, width, '
                        'crop_height, crop_width, '
                        'batch_size, '
                        'num_threads, '
                        'num_epochs, '
                        'do_stereo, '
                        'data_augment, '
                        'wrap_mode, '
                        'use_deconv, '
                        'alpha_image_loss, '
                        'disp_gradient_loss_weight, '
                        'lr_loss_weight, '
                        'full_summary')



    def test_dataloader(data_path, file_path, params, dataset, mode):
        with tf.Graph().as_default(), tf.device('/cpu:0'):
        #with tf.Graph().as_default():
    
            dataloader = StereoDataloader(data_path, file_path, params, dataset, mode)
            left = dataloader.left_image_batch
            right = dataloader.right_image_batch
            disp = dataloader.disp_image_batch
            print(left, right, disp)
    
            config = tf.ConfigProto(allow_soft_placement=False)
            sess = tf.Session(config=config)
    
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
    
            coordinator = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
    
            # test 2 batches for training
            for i in range(20):
                if mode == 'train' or mode == 'test':
                    img0, img1, d = sess.run([left, right, disp])
            
                    coordinator.request_stop()
                    coordinator.join(threads)
            
                    img0_b0 = img0[0,:,:,:]
                    img1_b0 = img1[0,:,:,:]
                    img0_b1 = img0[-1,:,:,:]
                    img1_b1 = img1[-1,:,:,:]
                    d_b0 = np.squeeze(d[0,:,:,:])
                    d_b1 = np.squeeze(d[-1,:,:,:])

                    mask1 = d_b0 > 0
                    mask2 = d_b1 > 0
                    print('batch: {}\nd0: {}\td1:{}\td0_min: {}\td0_max: {}\td1_min: {}\td1_max: {}'.format(i, 
                            np.sum(d_b0)/np.sum(mask1.astype(np.float32)),
                            np.sum(d_b1)/np.sum(mask2.astype(np.float32)),
                            np.min(d_b0),
                            np.max(d_b0),
                            np.min(d_b1),
                            np.max(d_b1)
))
                    print('img0: {}\timg1: {}'.format(np.mean(img0), np.mean(img1)))
                    print('shape: {}\t{}'.format(d.shape, d_b0.shape))
                    
                    
            
                    #plt.subplot(231)
                    #plt.imshow(img0_b0)
                    #plt.subplot(232)
                    #plt.imshow(img1_b0)
                    #plt.subplot(233)
                    #plt.imshow(d_b0)
                    #plt.subplot(234)
                    #plt.imshow(img0_b1)
                    #plt.subplot(235)
                    #plt.imshow(img1_b1)
                    #plt.subplot(236)
                    #plt.imshow(d_b1)
                    #plt.show()
                elif mode == 'inference':
                    img0, img1 = sess.run([left, right])
            
                    coordinator.request_stop()
                    coordinator.join(threads)
            
                    img0_b0 = img0[0,:,:,:]
                    img1_b0 = img1[0,:,:,:]
                    img0_b1 = img0[-1,:,:,:]
                    img1_b1 = img1[-1,:,:,:]
            
                    #plt.subplot(231)
                    #plt.imshow(img0_b0)
                    #plt.subplot(232)
                    #plt.imshow(img1_b0)
                    #plt.subplot(234)
                    #plt.imshow(img0_b1)
                    #plt.subplot(235)
                    #plt.imshow(img1_b1)
                    #plt.show()


            sess.close()
    
    data_path = '/BigDisk/fblu/flyingthings3d/'
    data_path = '/data/flyingthings3d/'
    file_path = './utils/filenames/flyingthings3d_train.txt'
    params = stereo_parameters(encoder = 'vgg',
                                  height = 540,
                                  width = 960,
                                  crop_height = 384,
                                  crop_width = 786,
                                  batch_size = 4,
                                  num_threads = 4,
                                  num_epochs = 1,
                                  do_stereo = True,
                                  data_augment = True,
                                  wrap_mode = 'border',
                                  use_deconv = True,
                                  alpha_image_loss = 0.1,
                                  disp_gradient_loss_weight = 0.1,
                                  lr_loss_weight = 0.1,
                                  full_summary = True)
    test_dataloader(data_path, file_path, params, 'flyingthings3d', 'train')
    print('flyingthings3d train: OK')

    data_path = '/BigDisk/fblu/flyingthings3d/'
    data_path = '/data/flyingthings3d/'
    file_path = './utils/filenames/flyingthings3d_test.txt'
    params = stereo_parameters(encoder = 'vgg',
                                  height = 540, 
                                  width = 960,
                                  crop_height = 384,
                                  crop_width = 786,
                                  batch_size = 4,
                                  num_threads = 1,
                                  num_epochs = 1,
                                  do_stereo = True,
                                  data_augment = False,
                                  wrap_mode = 'border',
                                  use_deconv = True,
                                  alpha_image_loss = 0.1,
                                  disp_gradient_loss_weight = 0.1,
                                  lr_loss_weight = 0.1,
                                  full_summary = True)
    test_dataloader(data_path, file_path, params, 'flyingthings3d', 'test')
    print('flyingthings3d test: OK')

    exit()

    data_path = '/data/stereo/data_scene_flow/'
    file_path = './utils/filenames/kitti_stereo_2015_train160_files.txt'
    params = stereo_parameters(encoder = 'vgg',
                                  height = 384, 
                                  width = 1242,
                                  crop_height = 256,
                                  crop_width = 512,
                                  batch_size = 4,
                                  num_threads = 2,
                                  num_epochs = 1,
                                  do_stereo = True,
                                  data_augment = True,
                                  wrap_mode = 'border',
                                  use_deconv = True,
                                  alpha_image_loss = 0.1,
                                  disp_gradient_loss_weight = 0.1,
                                  lr_loss_weight = 0.1,
                                  full_summary = True)
    test_dataloader(data_path, file_path, params, 'kitti', 'train')
    print('kitti train: OK')

    data_path = '/data/stereo/data_scene_flow/'
    file_path = './utils/filenames/kitti_stereo_2015_test_files.txt'
    params = stereo_parameters(encoder = 'vgg',
                                  height = 384, 
                                  width = 1242,
                                  crop_height = 384,
                                  crop_width = 1242,
                                  batch_size = 4,
                                  num_threads = 2,
                                  num_epochs = 1,
                                  do_stereo = True,
                                  data_augment = False,
                                  wrap_mode = 'border',
                                  use_deconv = True,
                                  alpha_image_loss = 0.1,
                                  disp_gradient_loss_weight = 0.1,
                                  lr_loss_weight = 0.1,
                                  full_summary = True)
    test_dataloader(data_path, file_path, params, 'kitti', 'inference')
    print('kitti inference: OK')
    

    
    
    
