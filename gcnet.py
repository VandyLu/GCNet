# batched image tensors should be (N, H, W, C)
import tensorflow as tf
import numpy as np
from collections import namedtuple

stereo_parameters = namedtuple('parameters', [
                        'height', 'width', 
                        'crop_height', 'crop_width', 
                        'batch_size', 
                        'num_threads', 
                        'num_epochs', 
                        'do_stereo', 
                        'data_augment', 
                        'alpha1',
                        'alpha2',
                        'alpha3',
                        'disp_gradient_loss_weight',
                        'lr_loss_weight',
                        'MDH_loss_weight',
                        'full_summary'])


batch_norm = tf.contrib.layers.batch_norm
convolution = tf.contrib.layers.convolution2d
deconv2d = tf.contrib.layers.convolution2d_transpose
deconv3d = tf.contrib.layers.convolution3d_transpose
initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)


class gcnet(object):
    def __init__(self, 
                 params,
                 img0,
                 img1,
                 disp,
                 max_disp = 192, 
                 base_num_filters = 32, 
                 first_kernel_size = 5, 
                 kernel_size = 3, 
                 num_res = 8, 
                 num_down_conv = 4, 
                 ds_stride = 2):

        self.params = params
        self.max_disp = max_disp
        self.base_num_filters = base_num_filters
        self.first_kernel_size = first_kernel_size
        self.kernel_size = kernel_size
        self.num_res = num_res
        self.num_down_conv = num_down_conv
        self.ds_stride = ds_stride
        self.img0 = img0
        self.img1 = img1
        self.disp = disp

        self.build_model()
        self.build_loss()
        self.build_summary()
    
    def conv2d_bn_relu(self, inputs, filters, kernel_size, stride, scope, reuse):
        return convolution(inputs, filters, kernel_size, stride = stride, activation_fn = tf.nn.relu, normalizer_fn = batch_norm, padding = 'SAME', weights_initializer = initializer, scope=scope, reuse = reuse)
    
    
    def residual_block(self, inputs, filters, kernel_size, stride, scope, reuse):
        with tf.variable_scope('res_a') as scope:
            f1 = self.conv2d_bn_relu(inputs, filters, kernel_size, stride, scope=scope, reuse = reuse)
        with tf.variable_scope('res_b') as scope:
            f2 = self.conv2d_bn_relu(f1, filters, kernel_size, stride, scope=scope, reuse = reuse)
        outputs = f2 + inputs
        return outputs
    
    def unary_features(self, inputs, filters, first_kernel_size, kernel_size, num_res, reuse):
        with tf.variable_scope('pre_conv') as scope: 
            f = self.conv2d_bn_relu(inputs, filters, first_kernel_size, 2, scope=scope, reuse=reuse)
    
        for i in range(num_res):
            with tf.variable_scope('res_{}'.format(i)) as scope:
        	f = self.residual_block(f, filters, kernel_size, 1, scope=scope, reuse=reuse)
    
        with tf.variable_scope('post_conv') as scope:
            outputs = convolution(f, filters, kernel_size, stride = 1, padding = 'SAME', normalizer_fn = None, activation_fn = None, weights_initializer = initializer, scope=scope, reuse = reuse)
        return outputs
    
    def get_cost_volume(self, left_feature, right_feature, max_disp, feature_size):
        shape = tf.shape(right_feature) 
        tshape = left_feature.shape
    
        # pads max_disp zeros on the left of each column
        paddings = [[0, 0], [0, 0], [max_disp, 0], [0, 0]]
        right_feature = tf.pad(right_feature, paddings)
        disparity_costs = []
        for d in reversed(range(max_disp)):
        	left_slice = left_feature
        	right_slice = tf.slice(right_feature, begin=[0, 0, d, 0], size=[-1, -1, shape[2], -1])
        	cost = tf.concat([left_slice, right_slice], axis = 3)
        	disparity_costs.append(cost)
    
        cost_volume = tf.stack(disparity_costs, axis=1)
        cost_volume.set_shape([tshape[0].value, max_disp, tshape[1].value, tshape[2].value, feature_size * 2])
        return cost_volume # NDHWC 5D tensor
    
    def conv3d_bn_relu(self, inputs, filters, kernel_size, stride, scope, reuse):
        return convolution(inputs, filters, kernel_size, stride, padding='SAME', data_format='NDHWC', activation_fn=tf.nn.relu, normalizer_fn=batch_norm, weights_initializer=initializer, scope=scope, reuse=reuse)
    
    def conv3d_down_sampling(self, inputs, filters, kernel_size, ds_stride, reuse):
        with tf.variable_scope('down_conv') as scope:
            down_conv = self.conv3d_bn_relu(inputs, filters, kernel_size, ds_stride, scope, reuse)
        with tf.variable_scope('conva') as scope:
            conv = self.conv3d_bn_relu(down_conv, filters, kernel_size, 1, scope, reuse)
        with tf.variable_scope('convb') as scope:
            conv = self.conv3d_bn_relu(conv, filters, kernel_size, 1, scope, reuse)
        return down_conv, conv
    
    def deconv3d_bn_relu(self, inputs, filters, kernel_size, stride, scope, reuse):
        return deconv3d(inputs, filters, kernel_size, stride=stride, activation_fn=tf.nn.relu, normalizer_fn=batch_norm, scope=scope, reuse=reuse) 
    
    def learning_regularization(self, cost_volume, base_num_filters, kernel_size, ds_stride, num_down_conv, reuse):
        res_list = []
        with tf.variable_scope('pre_conv3d'):
            with tf.variable_scope('conv') as scope: 
                conv = self.conv3d_bn_relu(cost_volume, base_num_filters, kernel_size, 1, scope, reuse)
            with tf.variable_scope('res') as scope: 
                res = self.conv3d_bn_relu(conv, base_num_filters, kernel_size, 1, scope, reuse)

        res_list.insert(0, res)
        down_conv = cost_volume
        for i in range(num_down_conv):
            if i < num_down_conv-1:
                mult = 2
            else:
        	mult = 4
            with tf.variable_scope('conv3d_{}'.format(i)) as scope:
                down_conv, res = self.conv3d_down_sampling(down_conv, mult*base_num_filters, kernel_size, ds_stride, reuse)

            res_list.insert(0, res)
    
        up_conv = res
        for i in range(num_down_conv):
            filters = res_list[i+1].shape[-1].value
            with tf.variable_scope('deconv3d_{}'.format(i)) as scope:
                deconv = self.deconv3d_bn_relu(up_conv, filters, kernel_size, ds_stride, scope, reuse)
            up_conv = deconv + res_list[i+1]

        # last layer, no BN or ReLu, filters = 1
        with tf.variable_scope('post_deconv3d') as scope:
            cv = deconv3d(up_conv, 1, kernel_size, stride=ds_stride, padding='SAME', activation_fn=None, normalizer_fn=None, scope=scope, reuse=reuse)
        cv = tf.squeeze(cv, axis=-1)
        return cv # NDHW
    
    def soft_argmin(self, cv, max_disp):
        softmin = tf.nn.softmax(-cv, dim=1) # NDHW
        softmin = tf.transpose(softmin, [0, 2, 3, 1]) # NHWD
        disp = tf.reshape(tf.range(max_disp, dtype=tf.float32), (1, 1, max_disp, 1))
        disp_map = tf.nn.conv2d(softmin, disp, strides=[1,1,1,1], padding='SAME', data_format='NHWC')
        #disp_map = tf.nn.conv2d(softmin, disp, strides=[1,1,1,1], padding='SAME', data_format='NCHW')
        #disp_map = tf.transpose(disp_map, [0, 2, 3, 1]) # N H W C

        shape = tf.shape(softmin)
        N, H, W, D = shape[0], shape[1], shape[2], shape[3]
        disp_volume = tf.tile(tf.reshape(tf.range(max_disp, dtype=tf.float32), (1, 1, 1, max_disp)), (N, H, W, 1))
        var_volume = tf.pow(disp_volume - tf.tile(disp_map, (1, 1, 1, max_disp)), 2) # kernel --> (X-E(X))^2
        var_map = tf.reduce_sum(var_volume * softmin, 3)
        #var_map = tf.nn.conv2d(softmin, var_kernel, strides=[1,1,1,1], padding='SAME', data_format='NCHW')
        #var_map = tf.transpose(var_map, [0, 2, 3, 1]) # N H W C

        entropy = -tf.reduce_sum(softmin * tf.log(softmin), 3)

        return disp_map, softmin, var_map, entropy

    def build_model(self):
        left_img = self.img0
        right_img = self.img1
    
        with tf.variable_scope('gc_net'):
            with tf.variable_scope('unary') as scope:
                # weight sharing 
                l_feature = self.unary_features(left_img, self.base_num_filters, self.first_kernel_size, self.kernel_size, self.num_res, reuse=False)
                r_feature = self.unary_features(right_img, self.base_num_filters, self.first_kernel_size, self.kernel_size, self.num_res, reuse=True)
        
            with tf.variable_scope('cost'):
                cost_volume = self.get_cost_volume(l_feature, r_feature, int(self.max_disp/2), self.base_num_filters)
            
            with tf.variable_scope('regularization'):
                cv = self.learning_regularization(cost_volume, self.base_num_filters, self.kernel_size, self.ds_stride, self.num_down_conv, reuse=False)
                self.cost_volume = cv
                
            with tf.variable_scope('soft_argmin'):
                disp_est, conf_map, var_map, entropy_map = self.soft_argmin(cv, self.max_disp)

        self.disp_est = disp_est
        self.conf_map = conf_map
        self.var_map = var_map
        self.entropy_map = entropy_map

        

    def build_loss(self):
        with tf.variable_scope('loss'):
            self.valid_mask = tf.cast(self.disp > 0, tf.float32)
            self.l1_loss_map = self.valid_mask * tf.abs(self.disp_est - self.disp)
            self.l2_loss_map = tf.pow(self.l1_loss_map, 2)
            
            self.count = tf.reduce_sum(self.valid_mask)
            self.l1_loss = tf.reduce_sum(self.l1_loss_map) / self.count
            self.l2_loss = tf.reduce_sum(self.l2_loss_map) / self.count


    def build_summary(self):
        with tf.variable_scope('summary'):
            self.summary_list = []
            self.summary_list.append(tf.summary.scalar('l1_loss', self.l1_loss))
            self.summary_list.append(tf.summary.scalar('l2_loss', self.l2_loss))
    
            if self.params.full_summary:
                self.summary_list.append(tf.summary.image('img0', self.img0))
                self.summary_list.append(tf.summary.image('img1', self.img1))
                self.summary_list.append(tf.summary.image('disp', self.disp))
                self.summary_list.append(tf.summary.image('disp_est', self.disp_est))
    
            self.summary_op = tf.summary.merge(self.summary_list)
            
    
    
