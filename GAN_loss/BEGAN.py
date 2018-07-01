from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np

from ops import *
from utils import *

class BEGAN(object):
    model_name = "BEGAN"     # name for checkpoint

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size

        if dataset_name == 'mnist' or dataset_name == 'fashion-mnist':
            # parameters
            self.input_height = 28
            self.input_width = 28
            self.output_height = 28
            self.output_width = 28

            self.z_dim = z_dim         # dimension of noise-vector
            self.c_dim = 1

            # BEGAN Parameter
            self.gamma = 0.75
            self.lamda = 0.001

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load mnist
            self.data_X, self.data_y = load_mnist(self.dataset_name)

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size

        elif dataset_name == 'img_align_celeba':
            # parameters
            self.input_height = 64
            self.input_width = 64
            self.output_height = 64
            self.output_width = 64

            self.z_dim = z_dim  # dimension of noise-vector
            self.c_dim = 3

            # BEGAN Parameter
            self.gamma = 0.75
            self.compare_gamma = 0
            self.lamda = 0.001

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 4  # number of generated images to be saved

            # get number of batches for a single epoch
            self.num_batches = calc_batch_num(self.dataset_name, batch_size=self.batch_size)
        else:
            raise NotImplementedError

    def discriminator(self, x, is_training=True, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):

            net = tf.nn.relu(conv2d(x, 64, 4, 4, 2, 2, name='en_conv1'))
            net = tf.nn.relu(bn(conv2d(x, 128, 4, 4, 2, 2, name='en_conv2'), is_training=is_training, scope='en_bn1'))
            net = tf.nn.relu(bn(conv2d(x, 256, 4, 4, 2, 2, name='en_conv3'), is_training=is_training, scope='en_bn2'))
            net = tf.reshape(net, [self.batch_size, -1])
            
            code = tf.nn.relu(bn(linear(net, 128, scope='d_fc6'), is_training=is_training, scope='d_bn6'))
            
            net = tf.nn.relu(bn(linear(code, 256 * 8 * 8, scope='de_fc1'), is_training=is_training, scope='de_bn1'))
            net = tf.reshape(net, [self.batch_size, 8, 8, 256])
            net = tf.nn.relu(bn(deconv2d(net, [self.batch_size, 16, 16, 128], 4, 4, 2, 2, name='de_dc1'), is_training=is_training, scope='de_bn2'))
            net = tf.nn.relu(bn(deconv2d(net, [self.batch_size, 32, 32, 64], 4, 4, 2, 2, name='de_dc2'), is_training=is_training, scope='de_bn3'))

            out = tf.nn.tanh(deconv2d(net, [self.batch_size, 64, 64, 3], 4, 4, 2, 2, name='de_dc3'))

            recon_error = tf.sqrt(2 * tf.nn.l2_loss(out - x)) / self.batch_size   # L1 norm
            return out, recon_error, code

    def generator(self, z, is_training=True, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):

            gf_dim = 128
            g0 = tf.reshape(linear(z, gf_dim * 8 * 4 * 4, scope='g_fc1'), [-1, 4, 4, gf_dim*8])
            g0 = tf.nn.relu(bn(g0, is_training=is_training, scope='g_bn0'))
            g1 = tf.nn.relu(bn(deconv2d(g0, [self.batch_size, 8, 8, gf_dim * 4], 4, 4, 2, 2, name='g_dc1'), is_training=is_training,
                               scope='g_bn1'))
            g2 = tf.nn.relu(bn(deconv2d(g1, [self.batch_size, 16, 16, gf_dim * 2], 4, 4, 2, 2, name='g_dc2'), is_training=is_training,
                               scope='g_bn2'))
            g3 = tf.nn.relu(bn(deconv2d(g2, [self.batch_size, 32, 32, gf_dim], 4, 4, 2, 2, name='g_dc3'), is_training=is_training,
                               scope='g_bn3'))
            g4 = tf.nn.tanh(deconv2d(g3, [self.batch_size, 64, 64, 3], 4, 4, 2, 2, name='g_dc5'))
            return g4

    def build_model(self):
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size

        """ BEGAN variable initial k = 0 """
        self.k = tf.Variable(0., trainable=False)

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')

        # noises
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')

        """ Loss Function """

        # output of D for real images
        D_real_img, D_real_err, D_real_code = self.discriminator(self.inputs, is_training=True, reuse=False)

        # output of D for fake images
        G = self.generator(self.z, is_training=True, reuse=False)
        D_fake_img, D_fake_err, D_fake_code = self.discriminator(G, is_training=True, reuse=True)

        # get loss for discriminator
        self.d_loss = D_real_err - self.k*D_fake_err
        self.compare_gamma = D_fake_err / D_real_err

        # get loss for generator
        self.g_loss = D_fake_err

        # convergence metric
        self.M = D_real_err + tf.abs(self.gamma*D_real_err - D_fake_err)

        # operation for updating k use tf.assign
        self.update_k = self.k.assign(
            tf.clip_by_value(self.k + self.lamda*(self.gamma*D_real_err - D_fake_err), 0, 1))

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                      .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1) \
                      .minimize(self.g_loss, var_list=g_vars)

        """" Testing """
        # for test
        self.fake_images = self.generator(self.z, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_error_real", D_real_err)
        d_loss_fake_sum = tf.summary.scalar("d_error_fake", D_fake_err)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        M_sum = tf.summary.scalar("M", self.M)
        k_sum = tf.summary.scalar("k", self.k)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
        self.p_sum = tf.summary.merge([M_sum, k_sum])

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim))

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
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

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                if self.dataset_name == 'mnist' or self.dataset_name == 'fashion-mnist':
                    batch_images = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                else:
                    batch_images = batch_loading_data(self.dataset_name, batch_size=self.batch_size, start_batch_id=idx, resize=True)
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                # update D network
                _, summary_str, d_loss, compare_gamma = self.sess.run([self.d_optim, self.d_sum, self.d_loss, self.compare_gamma],
                                               feed_dict={self.inputs: batch_images, self.z: batch_z})

                self.writer.add_summary(summary_str, counter)

                # update G network
                _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss], feed_dict={self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # update k
                if compare_gamma < self.gamma:
                    _, summary_str, M_value, k_value = self.sess.run([self.update_k, self.p_sum, self.M, self.k], feed_dict={self.inputs: batch_images, self.z: batch_z})
                    self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, M: %.8f, k: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss, M_value, k_value))

                # save training results for every 300 steps
                if np.mod(counter, 300) == 0:
                    samples = self.sess.run(self.fake_images, feed_dict={self.z: self.sample_z})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                './' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
                                    epoch, idx))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

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
        import re
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