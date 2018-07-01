from __future__ import division
import os
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import os, gzip
from PIL import Image
import tensorflow as tf
import tensorflow.contrib.slim as slim

# file_path = 'E:\Data\oriGAN'
file_path = '/home/jy/data_input'
# TF_path = 'E:\Data\mnist'
TF_path = '/home/jy/data_output/GAN/img'

def image_to_tfrecords(dataset_name, resize=False, re_h=128, re_w=128):
    data_dir = os.path.join(file_path, dataset_name)
    writer = tf.python_io.TFRecordWriter(TF_path + ".tfrecords")
    print("[*] Start writing...")
    print("Total images: %d" % len(os.listdir(data_dir)))
    count=0
    for img_name in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_name)
        img = Image.open(img_path)
        if resize:
            img = img.resize((re_h, re_w))
        imgb = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[imgb]))
        }))
        writer.write(example.SerializeToString())
        if count%10000 == 0:
            print(count, '/', len(os.listdir(data_dir)))
        count+=1
    writer.close()
    print("[*] Finish")

def imageL_to_tfrecords(dataset_name, classes, resize=False, re_h=128, re_w=128):
    data_dir = os.path.join(file_path, dataset_name)
    writer = tf.python_io.TFRecordWriter(file_path + dataset_name + ".tfrecords")
    for index, name in enumerate(classes):
        class_path = data_dir + name + '\\'
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            if resize:
                img = img.resize((re_h, re_w))
            imgb = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[imgb]))
            }))
            writer.write(example.SerializeToString())
    writer.close()


def load_tfrecords(dataset_name, labels=False):
    tf_dir = os.path.join("/home/jy/data_output/GAN", dataset_name + ".tfrecords")
    filename_queue = tf.train.string_input_producer([tf_dir])
    print("[*] Start reading...")
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    if labels:
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'img_raw': tf.FixedLenFeature([], tf.string)
                                           })
        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img = tf.reshape(img, [28, 28, 3])
        img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
        label = tf.cast(features['label'], tf.int32)
        return img, label
    else:
        record_iterator = tf.python_io.tf_record_iterator(path=tf_dir)
        all_img = tf.zeros([1, 28, 28, 3])
        count = 0
        print("[*] Start concat...")
        for string_record in record_iterator:
            features = tf.parse_single_example(serialized_example,
                                               features={'img_raw': tf.FixedLenFeature([], tf.string)})
            img = tf.decode_raw(features['img_raw'], tf.uint8)
            img = tf.reshape(img, [1, 28, 28, 3])
            if count == 0:
                all_img = img
            else:
                all_img = tf.concat([all_img, img], axis = 0)
            count += 1
            if count % 10000 == 0:
                print(count, all_img.shape)
        print(all_img[0].shape)
        # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
        return all_img
def calc_batch_num(dataset_name, batch_size=100):
    data_dir = os.path.join(file_path, dataset_name)
    file_dir = np.array(os.listdir(data_dir))
    num_batches = len(file_dir) // batch_size
    return num_batches

def load_mnist(dataset_name):
    data_dir = os.path.join(file_path, dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    # data type: image[image_num, image_weight, image_height, image_channel]
    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))     #(60000, 784) -> (60000, 28, 28, 1)

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    # mix training data and testing data
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    # data shuffle
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    #???
    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0
    # use sigmoid in G so divide by 255
    # return X / 255., y_vec
    return (X / 127.5) - 1, y_vec

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    """Training variable and all variable(has global_step)
       slim.model_analyzer to analyze the variable by graph or vars"""
    model_vars = tf.trainable_variables()
    all_model_vars = tf.all_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def imread(path, grayscale = False):
    if grayscale:
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    """Crop the center of the image"""
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image)/127.5 - 1.

def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width, resize_height, resize_width, crop)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def inverse_transform(images):
    # return (images+1.)/2.
    # print("Before inverse:", images)
    return (images+1.)*127.5

def save_images(images, size, image_path):
    imgs = inverse_transform(images)
    # print("After inverse: ", imgs)
    return imsave(imgs, size, image_path)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

""" Drawing Tools """
def save_scattered_image(z, id, z_range_x, z_range_y, name='scattered_image.jpg'):
    N = 10
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
    plt.colorbar(ticks=range(N))
    axes = plt.gca()
    axes.set_xlim([-z_range_x, z_range_x])
    axes.set_ylim([-z_range_y, z_range_y])
    plt.grid(True)
    plt.savefig(name)

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)