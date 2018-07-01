from __future__ import division
import os
import re
import time
import tensorflow as tf
import numpy as np

from math import *

from tools.loadImg import *
from tools.ops import *
from tools.utils import *

class EBGAN(object):
    model_name = "EBGAN"

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size

        if dataset_name == 'mnist' or dataset_name == 'fashion-mnist':
            self.data_X, self.data_y = load_mnist(self.dataset_name)
            self.input_height = self.data_X.shape[1]
            self.input_width = self.data_X.shape[2]
            self.output_height = self.data_X.shape[1]
            self.output_width = self.data_X.shape[2]
            self.channel_dim = self.data_X.shape[3]
            self.z_dim = z_dim

            self.pt_loss_weight = 0.1
            self.margin = max(1,self.batch_size/64.)

            self.learning_rate = 0.0002
            self.beta1 = 0.5
            self.sample_num = 64

            self.num_batches = len(self.data_X) // self.batch_size
        elif dataset_name == 'human_face':
            self.resize = True
            self.input_height, self.input_width, self.channel_dim, num_data = data_info(self.dataset_name)
            if self.resize == True:
                self.input_height, self.input_width = 64, 64
            self.output_height = self.input_height
            self.output_width = self.input_width
            self.z_dim = z_dim

            self.pt_loss_weight = 0.1
            self.margin = max(1,self.batch_size/64.)

            self.learning_rate = 0.0002
            self.beta1 = 0.5

            self.sample_num = 4 

            self.num_batches = num_data // self.batch_size
        else:
            raise NotImplementedError

    def pullaway_loss(self, embeddings):
        """
        Pull Away loss calculation
        :param embeddings: The embeddings to be orthogonalized for varied faces. Shape [batch_size, embeddings_dim]
        :return: pull away term loss
        """
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        similarity = tf.matmul(
            normalized_embeddings, normalized_embeddings, transpose_b=True)
        batch_size = tf.cast(tf.shape(embeddings)[0], tf.float32)
        pt_loss = (tf.reduce_sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
        return pt_loss

    def discriminator(self, x, is_training=True, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            df_dim = 64
            h, w = self.input_height, self.input_width
            D0 = lrelu(conv2d(x, df_dim, name='d_D0_conv'))
            D1 = lrelu(bn(conv2d(D0, df_dim * 2, name='d_D1_conv'), 
                                    is_training=is_training, scope='d_bn1'))
            D2 = lrelu(bn(conv2d(D1, df_dim * 4, name='d_D2_conv'), 
                                    is_training=is_training, scope='d_bn2'))
            D3 = lrelu(bn(conv2d(D2, df_dim * 8, name='d_D3_conv'), 
                                    is_training=is_training, scope='d_bn3'))
            print(D3.shape)
            code = linear(tf.reshape(D3, [self.batch_size, -1]), 1, 'd_code_lin')
            print(code.shape)
            d3 = tf.reshape(linear(code, df_dim * 8 * (ceil(h/16) * ceil(w/16)), scope='d_d3_lin'), 
                            [-1, ceil(h/16), ceil(w/16), df_dim * 8])
            d3 = tf.nn.relu(bn(d3, is_training=is_training, scope='d_bn4'))
            d2 = tf.nn.relu(bn(deconv2d(d3, [self.batch_size, ceil(h/8), ceil(w/8), df_dim * 4], name='d_dc1'), 
                                is_training=is_training, scope='d_bn5'))
            d1 = tf.nn.relu(bn(deconv2d(d2, [self.batch_size, ceil(h/4), ceil(w/4), df_dim * 2], name='d_dc2'), 
                                is_training=is_training, scope='d_bn6'))
            d0 = tf.nn.relu(bn(deconv2d(d1, [self.batch_size, ceil(h/2), ceil(w/2), df_dim], name='d_dc3'), 
                                is_training=is_training, scope='d_bn7'))
            out = tf.nn.tanh(deconv2d(d0, [self.batch_size, ceil(h), ceil(w), self.channel_dim], name='d_dc4'))
            recon_error = tf.sqrt(2 * tf.nn.l2_loss(out - x)) / self.batch_size
            return out, recon_error, code


    def generator(self, z, is_training=True, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            gf_dim = 64
            h, w = self.input_height, self.input_width 
            g0 = tf.reshape(linear(z, gf_dim * 8 * (ceil(h/16) * ceil(w/16)), scope='g_fc1'), [-1, ceil(h/16), ceil(w/16), gf_dim * 8])
            g0 = tf.nn.relu(bn(g0, is_training=is_training, scope='g_bn0'))
            g1 = tf.nn.relu(bn(deconv2d(g0, [self.batch_size, ceil(h/8), ceil(w/8), gf_dim * 4], name='g_dc1'), 
                                is_training=is_training, scope='g_bn1'))
            g2 = tf.nn.relu(bn(deconv2d(g1, [self.batch_size, ceil(h/4), ceil(w/4), gf_dim * 2], name='g_dc2'), 
                                is_training=is_training, scope='g_bn2'))
            g3 = tf.nn.relu(bn(deconv2d(g2, [self.batch_size, ceil(h/2), ceil(w/2), gf_dim], name='g_dc3'), 
                                is_training=is_training, scope='g_bn3'))
            g4 = tf.nn.tanh(deconv2d(g3, [self.batch_size, ceil(h), ceil(w), self.channel_dim], name='g_dc4'))
            return g4


    def build_model(self):
        image_dims = [self.input_height, self.input_width, self.channel_dim]
        bs = self.batch_size
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')
        # EBGAN Algo
        D_real_img, D_real_err, D_real_code = self.discriminator(self.inputs, is_training=True, reuse=False)
        G = self.generator(self.z, is_training=True, reuse=False)
        D_fake_img, D_fake_err, D_fake_code = self.discriminator(G, is_training=True, reuse=True)

        self.d_loss = D_real_err + tf.maximum(self.margin - D_fake_err, 0)
        self.g_loss = D_fake_err + self.pt_loss_weight*self.pullaway_loss(D_fake_code)

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                      .minimize(self.d_loss, var_list=d_vars)

            self.g_optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1) \
                      .minimize(self.g_loss, var_list=g_vars)

        self.fake_images = self.generator(self.z, is_training=False, reuse=True)

        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        self.g_sum = tf.summary.merge([g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_sum])

    def train(self):

        tf.global_variables_initializer().run()

        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim))

        self.saver = tf.train.Saver()

        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            for idx in range(start_batch_id, self.num_batches):
                if self.dataset_name == 'mnist' or self.dataset_name == 'fashion-mnist':
                    batch_images = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                else:
                    batch_images = batch_loading_data(self.dataset_name, batch_size=self.batch_size, start_batch_id=idx, resize=self.resize)

                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss], feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss], feed_dict={self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                counter += 1
                # if idx % 100 == 0:
                print("Epoch:[%2d] Batches:[%4d/%4d] time: %4.4f d_loss: %.8f g_loss: %.8f" \
                        % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))

                if np.mod(counter, 300) == 0:
                    samples = self.sess.run(self.fake_images, feed_dict={self.z: self.sample_z})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))

                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(epoch, idx))

            start_batch_id = 0

            self.save(self.checkpoint_dir, counter)

            self.visualize_results(epoch)

        self.save(self.checkpoint_dir, counter)

    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0