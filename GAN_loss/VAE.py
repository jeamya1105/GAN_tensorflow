from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np

from ops import *
from utils import *

import prior_factory as prior

class VAE(object):
    model_name = "VAE"     # name for checkpoint

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
            self.channel_dim = 1

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load mnist
            self.data_X, self.data_y = load_mnist(self.dataset_name)

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size

        else:
            raise NotImplementedError

    def decoder(self, z, is_training=True, reuse=False):
        with tf.variable_scope("decoder", reuse=reuse):

            de_dim = 128
            de0 = tf.reshape(linear(z, de_dim*8*2*2, scope='de_fc1'), [-1, 2, 2, de_dim*8])
            de0 = tf.nn.relu(bn(de0, is_training=is_training, scope='de_bn0'))
            de1 = tf.nn.relu(bn(deconv2d(de0, [self.batch_size, 4, 4, de_dim * 4], name='de_dc1'), is_training=is_training,
                               scope='de_bn1'))
            de2 = tf.nn.relu(bn(deconv2d(de1, [self.batch_size, 7, 7, de_dim * 2], name='de_dc2'), is_training=is_training,
                               scope='de_bn2'))
            de3 = tf.nn.relu(bn(deconv2d(de2, [self.batch_size, 14, 14, de_dim], name='de_dc3'), is_training=is_training,
                               scope='de_bn3'))
            de4 = tf.nn.sigmoid(deconv2d(de3, [self.batch_size, 28, 28, 1], name='de_dc4'))
            print(z.shape, de0.shape, de1.shape, de2.shape, de3.shape, de4.shape)
            return de4

    def encoder(self, x, is_training=True, reuse=False):
        with tf.variable_scope("encoder", reuse=reuse):

            en_dim = 128
            en0 = lrelu(conv2d(x, en_dim, name='en_d0_conv'))
            en1 = lrelu(bn(conv2d(en0, en_dim * 2, name='en_d1_conv'), is_training=is_training, scope='en_bn1'))
            en2 = lrelu(bn(conv2d(en1, en_dim * 4, name='en_d2_conv'), is_training=is_training, scope='en_bn2'))
            en3 = lrelu(bn(conv2d(en2, en_dim * 8, name='en_d3_conv'), is_training=is_training, scope='en_bn3'))
            # en4 = linear(tf.reshape(en3, [-1, en_dim * 8 * 2 * 2]), 1, 'en_d4_lin')
            en4 = lrelu(bn(linear(tf.reshape(en3, [-1, en_dim * 8 * 2 * 2]), en_dim*8*2*2, scope='en_ln4'), is_training=is_training, scope='en_bn4'))

            gaussian_params = linear(en4, 2 * self.z_dim, scope='en_fc4')
            mean = gaussian_params[:, :self.z_dim]
            stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, self.z_dim:])
            print(x.shape, en0.shape, en1.shape, en2.shape, en3.shape, en4.shape)
            return mean, stddev

    def build_model(self):
        # some parameters
        image_dims = [self.input_height, self.input_width, self.channel_dim]
        bs = self.batch_size

        """ Graph Input """
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')
        self.mu, sigma = self.encoder(self.inputs, is_training=True, reuse=False)
        z = self.mu + sigma * tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)

        out = self.decoder(z, is_training=True, reuse=False)
        self.out = tf.clip_by_value(out, 1e-8, 1 - 1e-8)

        marginal_likelihood = tf.reduce_sum(self.inputs * tf.log(self.out) + (1 - self.inputs) * tf.log(1 - self.out),
                                            [1, 2])
        KL_divergence = 0.5 * tf.reduce_sum(tf.square(self.mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, [1])

        self.neg_loglikelihood = -tf.reduce_mean(marginal_likelihood)
        self.KL_divergence = tf.reduce_mean(KL_divergence)

        ELBO = -self.neg_loglikelihood - self.KL_divergence

        self.loss = -ELBO

        t_vars = tf.trainable_variables()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1) \
                      .minimize(self.loss, var_list=t_vars)

        self.fake_images = self.decoder(self.z, is_training=False, reuse=True)


        nll_sum = tf.summary.scalar("nll", self.neg_loglikelihood)
        kl_sum = tf.summary.scalar("kl", self.KL_divergence)
        loss_sum = tf.summary.scalar("loss", self.loss)

        # final summary operations
        self.merged_summary_op = tf.summary.merge_all()

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = prior.gaussian(self.batch_size, self.z_dim)

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
                batch_images = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_z = prior.gaussian(self.batch_size, self.z_dim)

                # update autoencoder
                _, summary_str, loss, nll_loss, kl_loss = self.sess.run([self.optim, self.merged_summary_op, self.loss, self.neg_loglikelihood, self.KL_divergence],
                                               feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, nll: %.8f, kl: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, loss, nll_loss, kl_loss))

                # save training results for every 300 steps
                if np.mod(counter, 300) == 0:
                    samples = self.sess.run(self.fake_images,
                                            feed_dict={self.z: self.sample_z})

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

        z_sample = prior.gaussian(self.batch_size, self.z_dim)

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(
                        self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

        """ learned manifold """
        if self.z_dim == 2:
            assert self.z_dim == 2

            z_tot = None
            id_tot = None
            for idx in range(0, 100):
                #randomly sampling
                id = np.random.randint(0,self.num_batches)
                batch_images = self.data_X[id * self.batch_size:(id + 1) * self.batch_size]
                batch_labels = self.data_y[id * self.batch_size:(id + 1) * self.batch_size]

                z = self.sess.run(self.mu, feed_dict={self.inputs: batch_images})

                if idx == 0:
                    z_tot = z
                    id_tot = batch_labels
                else:
                    z_tot = np.concatenate((z_tot, z), axis=0)
                    id_tot = np.concatenate((id_tot, batch_labels), axis=0)

            save_scattered_image(z_tot, id_tot, -4, 4, name=check_folder(
                self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_learned_manifold.png')

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