import os, cv2
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

file_path = '/home/jy/data_input'

def data_info(dataset_name, multi_floder=False):
    if multi_floder == False:
        dataset_path = os.path.join('/home/jy/data_input', dataset_name)
        img = cv2.imread(os.path.join(dataset_path, os.listdir(dataset_path)[0]))
        data_size = len(os.listdir(dataset_path))
        return img.shape[0], img.shape[1], img.shape[2], data_size

def batch_loading_data(dataset_name, batch_size=100, start_batch_id=0, resize=False, re_h=64, re_w=64):
    data_dir = os.path.join(file_path, dataset_name)
    file_dir = np.array(os.listdir(data_dir))
    images = []
    for filename in file_dir[start_batch_id*batch_size:(start_batch_id+1)*batch_size]:
        image = plt.imread(os.path.join(data_dir, filename))
        if resize==True:
            image = scipy.misc.imresize(image, [re_h, re_w])
        image = image /127.5 - 1.
        image = image.tolist()
        images.append(image)
    return np.array(images)
