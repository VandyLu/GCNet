# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import numpy as np
import argparse
import re
import time
import tensorflow as tf

from gcnet import *
from stereo_dataloader import *

parser = argparse.ArgumentParser(description='Unsupervised stereo TensorFlow implementation.')

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='gcnet')
parser.add_argument('--supervised',                            help='has disparity or not', action='store_true')
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti, or cityscapes', default='kitti')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--data_augment',                          help='do data augmentation or not', action='store_true')
parser.add_argument('--input_height',              type=int,   help='input height', required=True)
parser.add_argument('--input_width',               type=int,   help='input width', required=True)
parser.add_argument('--crop_height',               type=int,   help='input height', default=256)
parser.add_argument('--crop_width',                type=int,   help='input width', default=512)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=1)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-3)
parser.add_argument('--lr_loss_weight',            type=float, help='left-right consistency weight', default=1.0)
parser.add_argument('--alpha1',                    type=float, help='weight of SSIM', default=0.8)
parser.add_argument('--alpha2',                    type=float, help='weight of L1 image loss', default=0.15)
parser.add_argument('--alpha3',                    type=float, help='weight of L1 Gradient loss', default=0.15)
parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.001)
parser.add_argument('--MDH_loss_weight',           type=float, help='weight of MDH loss', default=0.001)
parser.add_argument('--do_stereo',                             help='if set, will train the stereo model', action='store_true')
parser.add_argument('--warp_mode',                 type=str,   help='bilinear sampler wrap mode, edge or border', default='border')
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=4)
parser.add_argument('--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='')
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--full_summary',                          help='if set, will keep more data for each summary. Warning: the file can become very large', action='store_true')
parser.add_argument('--gpu',                       type=int,   help='specify gpu', default=0)

args = parser.parse_args()

def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

def train(params):
    """Training loop."""

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        global_step = tf.Variable(0, trainable=False)

        # OPTIMIZER
        num_training_samples = count_text_lines(args.filenames_file)

        steps_per_epoch = np.ceil(num_training_samples / params.batch_size).astype(np.int32)
        num_total_steps = params.num_epochs * steps_per_epoch
        start_learning_rate = args.learning_rate

        #boundaries = [np.int32(5000)]
        #values = [args.learning_rate, args.learning_rate / 10]
        #learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        learning_rate = args.learning_rate # const lr

        opt_step = tf.train.RMSPropOptimizer(learning_rate)

        print("total number of samples: {}".format(num_training_samples))
        print("total number of steps: {}".format(num_total_steps))

        dataloader = StereoDataloader(args.data_path, args.filenames_file, params, args.dataset, args.mode, supervised=args.supervised)
        left  = dataloader.left_image_batch
        right = dataloader.right_image_batch
        disp = dataloader.disp_image_batch

        with tf.device('/gpu:%d' % 0):
            model = gcnet(params, left, right, disp)
            loss = model.l1_loss
            train_op = opt_step.minimize(loss, global_step=global_step)

        tf.summary.scalar('learning_rate', learning_rate, ['model_0'])
        tf.summary.scalar('loss', loss, ['model_0'])
        summary_op = tf.summary.merge_all('model_0')

        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)

        # SAVER
        summary_writer = tf.summary.FileWriter(args.log_directory + '/' + args.model_name, sess.graph)
        train_saver = tf.train.Saver()

        # COUNT PARAMS
        total_num_parameters = 0
        for variable in tf.trainable_variables():
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()
        print("number of trainable parameters: {}".format(total_num_parameters))

        # INIT
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # LOAD CHECKPOINT IF SET
        if args.checkpoint_path != '':
            train_saver.restore(sess, args.checkpoint_path.split(".")[0])
            print('Restore from: {}'.format(args.checkpoint_path))

            if args.retrain:
                sess.run(global_step.assign(0))

        # GO!
        start_step = global_step.eval(session=sess)
        start_time = time.time()
        for step in range(start_step, num_total_steps):
            before_op_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - before_op_time
            if step and step % 10 == 0:
                examples_per_sec = params.batch_size / duration
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / step - 1.0) * time_sofar
                print_string = 'batch {:>6} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(step, examples_per_sec, loss_value, time_sofar, training_time_left))
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)
            if step and step % 5000 == 0:
                train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=step)

        train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=num_total_steps)

def test(params):
    """Test function."""

    dataloader = StereoDataloader(args.data_path, args.filenames_file, params, args.dataset, args.mode, supervised=args.supervised)
    left  = dataloader.left_image_batch
    right = dataloader.right_image_batch

    model = gcnet(params, left, right)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    if args.checkpoint_path == '':
        restore_path = tf.train.latest_checkpoint(args.log_directory + '/' + args.model_name)
    else:
        restore_path = args.checkpoint_path.split(".")[0]
    train_saver.restore(sess, restore_path)

    num_test_samples = count_text_lines(args.filenames_file)

    print('now testing {} files'.format(num_test_samples))
    disparities    = np.zeros((num_test_samples, params.crop_height, params.crop_width), dtype=np.float32)
    for step in range(num_test_samples):
        disp = sess.run(model.disp_est_left[0])
        disparities[step] = disp[0].squeeze()

    print('done.')

    print('writing disparities.')
    if args.output_directory == '':
        output_directory = os.path.dirname(args.checkpoint_path)
        #output_directory = os.path.dirname(args.log_directory + '/' + args.model_name)
    else:
        output_directory = args.output_directory
    np.save(output_directory + '/disparities.npy',    disparities)

    print('done.')

def test_epe(params):
    dataloader = StereoDataloader(args.data_path, args.filenames_file, params, args.dataset, args.mode, supervised=args.supervised)
    left  = dataloader.left_image_batch
    right = dataloader.right_image_batch
    disp = dataloader.disp_image_batch

    assert params.crop_width == params.width and params.crop_height == params.height
    #assert params.height == 384 and params.width == 1280

    # split image to two parts due to limited GPU memory
    left_ph = tf.placeholder(tf.float32, [1, np.int32(params.height/2), np.int32(params.width), 3])
    right_ph = tf.placeholder(tf.float32, [1, np.int32(params.height/2), np.int32(params.width), 3])
    disp_ph = tf.placeholder(tf.float32, [1, np.int32(params.height/2), np.int32(params.width), 1])

    model = gcnet(params, left_ph, right_ph, disp_ph)
    mask = disp_ph > 0
    mask_map = tf.cast(mask, tf.float32)
    count = tf.reduce_sum(mask_map)

    epe_map = tf.abs(model.disp_est - disp_ph) * mask_map
    epe = tf.reduce_sum(epe_map) / count
    acc_map = tf.logical_and(tf.logical_or(epe_map < 3.0, epe_map/disp_ph < 0.05), mask)
    acc = tf.reduce_sum(tf.cast(acc_map, tf.float32)) / count

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    if args.checkpoint_path == '':
        restore_path = tf.train.latest_checkpoint(args.log_directory + '/' + args.model_name)
    else:
        restore_path = args.checkpoint_path.split(".")[0]
    train_saver.restore(sess, restore_path)
    print('Restore from: {}'.format(restore_path))

    num_test_samples = count_text_lines(args.filenames_file)

    print('now testing {} files'.format(num_test_samples))
    disparities    = np.zeros((num_test_samples, params.crop_height, params.crop_width), dtype=np.float32)
    variance       = np.zeros((num_test_samples, params.crop_height, params.crop_width), dtype=np.float32)
    entropy        = np.zeros((num_test_samples, params.crop_height, params.crop_width), dtype=np.float32)
    groundtruth    = np.zeros((num_test_samples, params.crop_height, params.crop_width), dtype=np.float32)

    total_epe_avg = 0.0
    total_acc_avg = 0.0
    half_height = np.int32(params.height/2)
    for step in range(num_test_samples):
        left_image, right_image, disp_image = sess.run([left, right, disp])
        
        left_0  =  left_image[:, 0:half_height, 0:params.width, :]
        right_0 = right_image[:, 0:half_height, 0:params.width, :]
        disp_0  =  disp_image[:, 0:half_height, 0:params.width, :]

        disp_0_value, var_0_value, entropy_0_value, epe_0_value, acc_0_value, count_0_value = sess.run([model.disp_est, model.var_map, model.entropy_map, epe, acc, count], feed_dict={left_ph: left_0, right_ph: right_0, disp_ph: disp_0})

        left_1  =  left_image[:, half_height:params.height, 0:params.width, :]
        right_1 = right_image[:, half_height:params.height, 0:params.width, :]
        disp_1  =  disp_image[:, half_height:params.height, 0:params.width, :]

        disp_1_value, var_1_value, entropy_1_value, epe_1_value, acc_1_value, count_1_value = sess.run([model.disp_est, model.var_map, model.entropy_map, epe, acc, count], feed_dict={left_ph: left_1, right_ph: right_1, disp_ph: disp_1})

        disp_value     = np.concatenate([disp_0_value, disp_1_value], 1)
        var_value      = np.concatenate([var_0_value, var_1_value], 1)
        entropy_value  = np.concatenate([entropy_0_value, entropy_1_value], 1)
        acc_value = (acc_0_value * count_0_value + acc_1_value * count_1_value) / (count_0_value + count_1_value)
        epe_value = (epe_0_value * count_0_value + epe_1_value * count_1_value) / (count_0_value + count_1_value)

        groundtruth[step] = disp_image[0].squeeze()
        disparities[step] = disp_value[0].squeeze()
        variance[step] = var_value[0].squeeze()
        entropy[step] = entropy_value[0].squeeze()
        print('step: {}\tepe: {}\tacc: {}'.format(step, epe_value, acc_value))
        
        total_epe_avg += epe_value
        total_acc_avg += acc_value

    total_epe_avg /= num_test_samples
    total_acc_avg /= num_test_samples
    print('total_epe: {}\ttotal_avg: {}'.format(total_epe_avg, total_acc_avg))

    print('done.')

    print('writing disparities.')
    if args.output_directory == '':
        output_directory = os.path.dirname(args.checkpoint_path)
        #output_directory = os.path.dirname(args.log_directory + '/' + args.model_name)
    else:
        output_directory = args.output_directory
    np.save(output_directory + '/disparities.npy',    disparities)
    np.save(output_directory + '/variance.npy',       variance)
    np.save(output_directory + '/entropy.npy',        entropy)

    print('done.')


def main(_):

    params = stereo_parameters(
        height=args.input_height,
        width=args.input_width,
        crop_height=args.crop_height,
        crop_width=args.crop_width,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        num_epochs=args.num_epochs,
        do_stereo=args.do_stereo,
        data_augment=args.data_augment,
        alpha1=args.alpha1,
        alpha2=args.alpha2,
        alpha3=args.alpha3,
        disp_gradient_loss_weight=args.disp_gradient_loss_weight,
        lr_loss_weight=args.lr_loss_weight,
        MDH_loss_weight=args.MDH_loss_weight,
        full_summary=args.full_summary)

    if args.mode == 'train':
        train(params)
    elif args.mode == 'test':
        #test(params)
        test_epe(params)

if __name__ == '__main__':
    tf.app.run()
