import os, cv2, logging
import scipy.misc as sm

data_path = '/home/jy/data_input/cars'
sample = 2

class DS(object):

    def __init__(self, data_path, sample):
        logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

        self.data_path = data_path
        self.sample = sample

    def saveImg(self, save_path, img, name):
        cv2.imwrite(os.path.join(save_path, str(name)), img)

    def reSize(self, img_size=512):
         if not os.path.exists(os.path.join(self.data_path, 'resize_img')):
            os.makedirs(os.path.join(self.data_path, 'resize_img'))
         data_path = os.path.join(self.data_path, 'ori')
         save_path = os.path.join(self.data_path, 'resize_img')
         logging.info("Resize and Saving...")
         for i in os.listdir(data_path):
            img = cv2.imread(os.path.join(data_path, str(i)), -1)
            img = sm.imresize(img, [img_size, img_size, 3])
            self.saveImg(save_path, img, i)

         logging.info(" Finish ")


    def downSampling(self):
        if not os.path.exists(os.path.join(self.data_path, str(self.sample*2))):

            os.makedirs(os.path.join(self.data_path, str(self.sample*2)))

        self.save_path = os.path.join(self.data_path, str(self.sample*2))

        logging.info(" Downsampling and Saving ")

        self.data_path = os.path.join(data_path, 'resize_img')

        for i in os.listdir(self.data_path):

            img = cv2.imread(os.path.join(self.data_path, str(i)), -1)

            tmp = self.sample

            while tmp != 0:

                img = cv2.pyrDown(img)

                tmp -= 1

            self.saveImg(self.save_path, img, i)

        logging.info(" Finish ")


if __name__ == '__main__':
    DS = DS(data_path, sample)
    #DS.reSize()
    DS.downSampling()
