import os
import tensorflow as tf
import argparse

from GAN_loss.oriGAN import GAN
from GAN_loss.WGAN import WGAN
from GAN_loss.WGAN_GP import WGAN_GP
from GAN_loss.FisherGAN import FisherGAN
from GAN_loss.EBGAN import EBGAN
from GAN_loss.ACGAN import ACGAN
# from GAN_loss.VAE import VAE
# from GAN_loss.ACGAN import ACGAN
# from GAN_loss.BEGAN import BEGAN

from tools.utils import show_all_variables
from tools.utils import check_folder

def parse_args():
    desc = "Tensorflow implementation of GANs"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--gan_type', type=str, default='GAN',
                        choices=['GAN', 'WGAN', 'WGAN_GP', 'FisherGAN', 'VAE', 'ACGAN', 'EBGAN', 'BEGAN'], help='The type of GAN', required=True)
    parser.add_argument('--dataset',type=str, default='mnist', choices=['mnist', 'human_face', 'fashion-mnist', 'celebA', 'local_data'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=100, help='Dimension of noise vector')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    return check_args(parser.parse_args())

def check_args(args):
    # check folder
    check_folder(args.checkpoint_dir)
    check_folder(args.result_dir)
    check_folder(args.log_dir)

    # check hyperparameter
    assert args.epoch >= 1, 'epochs must >= 1 ...'
    assert args.batch_size >= 1, 'batch size must >= 1 ...'
    assert args.z_dim >= 1, 'dimension of noise vector must >= 1'
    
    return args

def main():
    # parse arguments
    args = parse_args()
    exit() if args is None else None
    # model = [GAN, CGAN, VAE, ACGAN, EBGAN, BEGAN]
    model = [ACGAN]

    # control GPU usage 70%
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options = gpu_options)) as sess:
        gan = None
        # if args.gan_type == model[0].model_name:
        gan = model[0](sess, 
                    epoch=args.epoch,
                    batch_size=args.batch_size,
                    z_dim=args.z_dim,
                    dataset_name=args.dataset,
                    checkpoint_dir=args.checkpoint_dir,
                    result_dir=args.result_dir,
                    log_dir=args.log_dir)
        if gan is None:
            raise Exception("[!] There is no option for " + args.gan_type)

        gan.build_model()
        #show_all_variables()
        gan.train()
        print(" [*] Training finished!")

        # visualize learned generator
        gan.visualize_results(args.epoch-1)
        print(" [*] Testing finished!")

if __name__ == '__main__':
    # use GPU 0
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()